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
import math
import argparse
import random
import time
import numpy as np
import torch

from lib.hiro.hiro_parking import HIROTrainer


def run_fun(seed_list,
            num_outer_steps,
            iter_size,
            max_action,
            batch_size,
            tau,
            expl_noise,
            policy_noise,
            noise_clip,
            policy_freq,
            max_iterations,
            start_time_steps,
            num_eval=100
            ):

    # INITIALIZATION
    start_time = time.time()
    prefix = 'hiro_parking'
    suffix = ''

    config_dict = {
        'prefix': prefix,
        'suffix': suffix,
        'barcode': int(start_time),
        'seed_list': seed_list,
        'num_outer_steps': num_outer_steps,
        'iter_size': iter_size,
        'max_action': max_action,
        'batch_size': batch_size,
        'tau': tau,
        'expl_noise': expl_noise,
        'policy_noise': policy_noise,
        'noise_clip': noise_clip,
        'policy_freq': policy_freq,
        'max_iterations': max_iterations,
        'start_time_steps': start_time_steps
    }
    save_path = ""

    f = open(save_path + "config_" + prefix + "_" + str(config_dict['barcode']) + "_" + suffix + '.txt', 'w')
    f.write(str(config_dict))
    f.close()

    print("CONFIG")
    print(config_dict)

    # LOOP OVER SEEDS #
    for seed_val in seed_list:

        # SEED #
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)

        # INITIALIZATION #
        myHIRO = HIROTrainer(iter_num=5,
                             iter_size=iter_size,
                             state_dim=12,
                             action_dim=3,
                             max_action=max_action,
                             batch_size=batch_size,
                             discount=0.99,
                             tau=tau,
                             expl_noise=expl_noise,
                             max_iter=max_iterations,
                             start_time_steps=start_time_steps,
                             policy_noise=policy_noise,
                             noise_clip=noise_clip,
                             policy_freq=policy_freq,
                             total_iter=num_outer_steps)
        eps_tot = 0
        acc_eps_list = [0]
        acc_result_list = [0]

        print("\n")
        print("SEED")
        print(seed_val)
        print("\n")

        i = 0
        while eps_tot < num_outer_steps:
            i += 1
            print(eps_tot)

            print("TRAIN")
            myHIRO.train()
            eps_tot += 5

            print("TEST")
            reach_prob = 0
            for ii in range(num_eval):
                g_success = myHIRO.test()
                reach_prob += g_success
            reach_prob = reach_prob / num_eval

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

    run_fun(seed_list=seed_l,
            num_outer_steps=300,
            iter_size=10000,
            max_action=np.array([0.04, 0.04, math.pi]),
            batch_size=256,
            tau=0.005,
            expl_noise=0.1,
            max_iterations=100,
            start_time_steps=1e4,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
            )
