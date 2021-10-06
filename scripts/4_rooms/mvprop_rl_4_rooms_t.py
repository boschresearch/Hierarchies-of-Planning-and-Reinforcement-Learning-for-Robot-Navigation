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


# IMPORT #
import argparse
import random
import time
import numpy as np
import torch

from lib.mvprop.mvprop_rl_4_rooms import TRPOTrainer


class CONFIG:
    def __init__(self,
                 seed_list,
                 time_horizon,
                 num_inner_episodes,
                 num_outer_episodes,
                 discount_factor,
                 policy_max_kl,
                 policy_damp_val,
                 policy_hidden_sizes,
                 policy_batch_size,
                 mvprop_l_h,
                 mvprop_k,
                 mvprop_buffer_size,
                 mvprop_lr,
                 mvprop_batch_size,
                 mvprop_decay_type,
                 mvprop_decay,
                 mvprop_target_update_frequency,
                 mvprop_her,
                 terrain_var,
                 num_eval
                 ):
        self.terrain_var = terrain_var
        self.mvprop_decay_type = mvprop_decay_type
        self.mvprop_her = mvprop_her
        self.mvprop_target_update_frequency = mvprop_target_update_frequency
        self.mvprop_decay = mvprop_decay
        self.num_eval = num_eval
        self.mvprop_batch_size = mvprop_batch_size
        self.mvprop_lr = mvprop_lr
        self.mvprop_buffer_size = mvprop_buffer_size
        self.mvprop_k = mvprop_k
        self.mvprop_l_h = mvprop_l_h
        self.policy_batch_size = policy_batch_size
        self.policy_hidden_sizes = policy_hidden_sizes
        self.policy_damp_val = policy_damp_val
        self.policy_max_kl = policy_max_kl
        self.discount_factor = discount_factor
        self.num_outer_episodes = num_outer_episodes
        self.num_inner_episodes = num_inner_episodes
        self.time_horizon = time_horizon
        self.seed_list = seed_list


def run_fun(config):

    # INITIALIZATION
    start_time = time.time()
    prefix = 'mvprop_rl_4_rooms_t'
    suffix = ''

    config_dict = config.__dict__
    config_dict['prefix'] = prefix
    config_dict['suffix'] = suffix
    config_dict['barcode'] = int(start_time)

    save_path = ""

    f = open(save_path + "config_" + prefix + "_" + str(config_dict['barcode']) + "_" + suffix + '.txt', 'w')
    f.write(str(config_dict))
    f.close()

    print("CONFIG")
    print(config_dict)

    # LOOP OVER SEEDS #
    for seed_val in config.seed_list:

        # SEED #
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)

        # INITIALIZATION #
        myTRPO = TRPOTrainer(env_mode='grid',
                             op_mode='mvprop_rl',
                             state_dim=12,
                             action_dim=2,
                             config=config
                             )

        eps_tot = 0
        acc_eps_list = [0]
        acc_result_list = [0]

        print("\n")
        print("SEED")
        print(seed_val)
        print("\n")

        i = 0
        while eps_tot < config.num_outer_episodes:
            i += 1
            print(eps_tot)

            print("TRAIN")
            myTRPO.train()
            eps_tot += config.num_inner_episodes

            print("TEST")
            reach_prob = 0
            for ii in range(config.num_eval):
                g_success = myTRPO.test()
                reach_prob += g_success
            reach_prob = reach_prob / config.num_eval

            print("Iteration: %i" % i)
            print("Reach Prob: %.2f" % reach_prob)
            print("Time passed (in min): %.2f" % ((time.time() - start_time) / 60))

            acc_eps_list.append(eps_tot)
            acc_result_list.append(reach_prob)

        # SAVE RESULTS #
        episodes_np = np.array(acc_eps_list)
        np.save(save_path + "episodes_" + prefix + "_" + str(seed_val) + "_" + suffix + "_"
                + str(config_dict['barcode']), episodes_np)

        np.save(save_path + "results_" + prefix + "_" + str(seed_val) + "_" + suffix + "_"
                + str(config_dict['barcode']), np.array(acc_result_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="s1")
    args = parser.parse_args()
    if args.seed == "s1":
        seed_l = [1535719580]
    elif args.seed == "s2":
        seed_l = [1535720536]
    elif args.seed == "s3":
        seed_l = [1535721129]
    elif args.seed == "s4":
        seed_l = [1535721985]
    elif args.seed == "s5":
        seed_l = [1535723522]
    elif args.seed == "s6":
        seed_l = [1535724275]
    elif args.seed == "s7":
        seed_l = [1535726291]
    elif args.seed == "s8":
        seed_l = [1535954757]
    elif args.seed == "s9":
        seed_l = [1535957367]
    elif args.seed == "s10":
        seed_l = [1535953242]
    else:
        seed_l = [1535719580, 1535720536, 1535721129, 1535721985, 1535723522,
                  1535724275, 1535726291, 1535954757, 1535957367, 1535953242]

    run_config = CONFIG(seed_list=seed_l,
                        time_horizon=100,
                        num_inner_episodes=5,
                        num_outer_episodes=250,
                        discount_factor=0.99,
                        policy_max_kl=5e-4,
                        policy_damp_val=5e-3,
                        policy_hidden_sizes=[64, 64, 64],
                        policy_batch_size=3200,
                        mvprop_l_h=150,
                        mvprop_k=100,
                        mvprop_buffer_size=35000,
                        mvprop_lr=3e-4,
                        mvprop_batch_size=128,
                        mvprop_decay=10000,
                        mvprop_her=True,
                        mvprop_target_update_frequency=1,
                        num_eval=100,
                        mvprop_decay_type='exp',
                        terrain_var=True)

    run_fun(run_config)
