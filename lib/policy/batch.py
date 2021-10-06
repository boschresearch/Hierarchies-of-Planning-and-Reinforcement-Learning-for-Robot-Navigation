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


from typing import Optional, List

from numpy import array, mean, ndarray


class Batch(dict):
    def __init__(self,
                 actions: Optional[list] = None,
                 states: Optional[List[ndarray]] = None,
                 rewards: Optional[list] = None,
                 next_states: Optional[List[ndarray]] = None,
                 masks: Optional[list] = None,
                 add_info=None,
                 weight=None,
                 num_episodes=None):
        """ 
        Stores data obtained by one or more rollout.

        The basic data container for class MakeRollout
        (from rlrl.training.makerollout) and derived. The data of trajectories
        obtained from rollouts is stored in the following way: The data of 
        individual trajectories is stacked into single arrays
        (
        self['actions'] = actions 
        self['states'] = states 
        self['rewards'] = rewards
        self['next_states'] = next_states
        self['masks'] = masks
        self['add_info'] = add_info
        ).
        The data of key 'masks' indicates the termination of the individual
        trajectories in the following way:
        self['masks'][i] = 1 if the state
        self['states'][i] is non-terminal,
        self['masks'][i] = 0 if the state
        self['states'][i] is terminal

        :param actions: array of actions
        :param states: array of states
        :param rewards: array of rewards
        :param next_states: array of successor states
        :param masks: array of mask elements indicating termination of a trajectory
        :param add_info: array of additional information for debugging
        """
        super(Batch, self).__init__()

        self['actions'] = actions if actions is not None else []
        self['states'] = states if states is not None else []
        self['rewards'] = rewards if rewards is not None else []
        self['next_states'] = next_states if next_states is not None else []
        self['masks'] = masks if masks is not None else []
        self['add_info'] = add_info if add_info is not None else []
        self['weight'] = weight if weight is not None else []
        self['num_episodes'] = num_episodes if num_episodes is not None else 0

        for obj in [self['actions'], self['states'], self['next_states'],
                    self['masks']]:
            assert isinstance(obj, list)

    def append(self, batch):
        for key, value in batch.items():
            try:
                if key == "num_episodes":
                    self[key] += value
                else:
                    self[key].extend(value)
            except Exception as e:
                print(f'Exception {e} for key={key} old value={self[key]} '
                      f'new value={value}|')
                raise

    def avg_return(self, discount=1):
        """ Calculates the average return per rollout in this batch. """
        returns = []
        ret = 0
        weight = 1
        for ix, r in enumerate(self['rewards']):
            # Accumulate reward into retrun
            ret += weight * r
            weight = weight * discount
            if self['masks'][ix] == 0:  # if trajectory terminates
                returns.append(ret)  # generate a new return sample
                ret = 0
        return mean(returns)  # take average over return samples

    def length(self):
        return len(self['actions'])

    def num_episodes(self):
        return (array(self['masks']) == 0).sum()
