# Hierarchies of Planning and Reinforcement Learning for Robot Navigation
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import math

import numpy as np
import scipy.optimize
import torch

from lib.policy import pytorch as torch_utils
from lib.policy.models.mlp_critic import StateValue
from lib.optimizers.actor_critic.advantages import gae
from lib.optimizers.base import Optimizer


class ActorCriticOptimizer(Optimizer):
    def __init__(self, policy, discount=0.998, gae_lambda=0.95, l2_reg=0, imp_weight=False,
                 **kwargs):
        super(ActorCriticOptimizer, self).__init__(policy, **kwargs)
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.l2_reg = l2_reg
        self.imp_weight = imp_weight
        print("DISCOUNT:")
        print(self.discount)
        self.isnan_count = 0

    def _update_networks(self, policy,
                         actions, masks, rewards, states, num_episodes, weights=None, *args):
        values = self.networks["critic"](states).detach()
        print(values.shape)
        advantages, returns = gae(
            rewards, masks, values, discount=self.discount,
            gae_lambda=self.gae_lambda, use_gpu=self.use_gpu)

        optimizer_str_list = [key for key in
                              ["policy", "critic"][:len(self.optimizers)]]
        optimizers = [self.optimizers[key] for key in optimizer_str_list]
        self.step(policy, self.networks["critic"], *optimizers[:2],
                  states, actions, returns, advantages, weights)
        return policy

    @classmethod
    def _init_networks(cls, obs_dim, action_dim):
        return {"critic": StateValue(obs_dim)}

    @staticmethod
    def step(*args):
        raise NotImplementedError

# The following class is derived from PyTorch-RL
#   (https://github.com/Khrylx/PyTorch-RL)
# Copyright (c) 2020 Ye Yuan, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree,
# which derived it from pytorch-trpo
#   (https://github.com/ikostrikov/pytorch-trpo)
# Copyright (c) 2017 Ilya Kostrikov, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

class TRPO(ActorCriticOptimizer):
    def __init__(self, policy, max_kl=1e-1, damping=1e-2, use_fim=True,
                 **kwargs):
        super(TRPO, self).__init__(policy, **kwargs)
        self.max_kl = max_kl
        self.damping = damping
        self.use_fim = use_fim

    def step(self, policy_net, value_net, states, actions, returns, advantages, weights=None):

        """update critic"""
        values_target = torch.clamp(returns, min=-10.0, max=10.0)

        def get_value_loss(flat_params):
            torch_utils.set_flat_params_to(value_net,
                                           torch_utils.torch.from_numpy(flat_params))
            for param in value_net.parameters():
                if param.grad is not None:
                    param.grad.detach().fill_(0)
            values_pred = value_net(states)
            values_pred = torch.clamp(values_pred, min=-10.0, max=10.0)
            value_loss = (values_pred - values_target).pow(2).mean()
            value_loss.backward()

            return value_loss.detach().cpu().item(), \
                   torch_utils.get_flat_grad_from(
                       value_net.parameters()).detach().cpu().numpy(). \
                       astype(np.float64)

        # value_net.train()
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
            get_value_loss,
            torch_utils.get_flat_params_from(value_net).cpu().numpy(), maxiter=25)
        torch_utils.set_flat_params_to(value_net, torch.from_numpy(flat_params))

        """update policy"""
        states.requires_grad_(requires_grad=True)
        fixed_log_probs = policy_net.get_log_prob(states, actions).detach()

        """define the loss function for TRPO"""
        def get_loss(volatile=False):
            log_probs = policy_net.get_log_prob(states.clone().detach().requires_grad_(volatile), actions)
            action_loss = -(advantages * torch.exp(log_probs - fixed_log_probs)).mean()
            return action_loss

        def Fvp_fim(v):
            """
            use fisher information matrix for Hessian*vector
            """
            M, mu, info = policy_net.get_fim(states)
            mu = mu.view(-1)
            filter_input_ids = set() if policy_net.is_disc_action else \
                {info['std_id']}

            t = M.new(mu.size())
            t[:] = 1
            t.requires_grad_(requires_grad=True)
            mu_t = (mu * t).sum()
            Jt = torch_utils.compute_flat_grad(mu_t, policy_net.parameters(),
                                               filter_input_ids=filter_input_ids,
                                               create_graph=True)
            Jtv = (Jt * v).sum()
            Jv = torch.autograd.grad(Jtv, t, retain_graph=True)[0]
            MJv = M * Jv.detach()
            mu_MJv = (MJv * mu).sum()
            JTMJv = torch_utils.compute_flat_grad(mu_MJv,
                                                  policy_net.parameters(),
                                                  filter_input_ids=filter_input_ids,
                                                  retain_graph=True).data
            JTMJv /= states.shape[0]
            if not policy_net.is_disc_action:
                std_index = info['std_index']
                JTMJv[std_index: std_index + M.shape[0]] += \
                    2 * v[std_index: std_index + M.shape[0]]
            return JTMJv + v * self.damping

        def Fvp_direct(v):
            """
            directly compute Hessian*vector from KL
            """
            kl = policy_net.get_kl(states.clone().detach())
            kl = kl.mean()

            grads = torch.autograd.grad(kl, policy_net.parameters(),
                                        create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v.clone().detach()).sum()
            grads = torch.autograd.grad(kl_v, policy_net.parameters())
            flat_grad_grad_kl = torch.cat(
                [grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * self.damping

        Fvp = Fvp_fim if self.use_fim else Fvp_direct

        loss = get_loss()
        grads = torch.autograd.grad(loss, policy_net.parameters())
        loss_grad = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
        lm = math.sqrt(self.max_kl / shs)
        if np.isnan(lm):
            # print("LM ISNAN")
            # lm = 1.
            self.isnan_count += 1
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)

        prev_params = torch_utils.get_flat_params_from(policy_net)
        success, new_params = \
            line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
        torch_utils.set_flat_params_to(policy_net, new_params)

        return success

    @staticmethod
    def _init_optimizers(networks, lr_rates=None):
        return []

# The following fct is derived from PyTorch-RL
#   (https://github.com/Khrylx/PyTorch-RL)
# Copyright (c) 2020 Ye Yuan, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree,
# which derived it from pytorch-trpo
#   (https://github.com/ikostrikov/pytorch-trpo)
# Copyright (c) 2017 Ilya Kostrikov, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = b.clone()
    x[:] = 0
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x

# The following fct is derived from PyTorch-RL
#   (https://github.com/Khrylx/PyTorch-RL)
# Copyright (c) 2020 Ye Yuan, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree,
# which derived it from pytorch-trpo
#   (https://github.com/ikostrikov/pytorch-trpo)
# Copyright (c) 2017 Ilya Kostrikov, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10,
                accept_ratio=0.1):
    fval = f(True).detach()

    steps = [.5 ** x for x in range(max_backtracks)]
    for stepfrac in steps:
        x_new = x + stepfrac * fullstep
        try:
            torch_utils.set_flat_params_to(model, x_new)
            fval_new = f(True).detach()
            actual_improve = fval - fval_new
            expected_improve = expected_improve_full * stepfrac
            ratio = actual_improve / expected_improve

            if ratio > accept_ratio:
                return True, x_new
        except RuntimeError as e:
            print("Runtime Error {} Ignored! Stepsize Reduced".format(e))
    return False, x


def _calculate_state_actions(states, actions):
    return torch.tensor(
        torch.cat([states, actions.float().unsqueeze(-1)], dim=1))


def _calculate_surrogate_rewards(discriminator, policy_state_actions):
    return discriminator.surrogate_reward(policy_state_actions)
