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
#
# This source code is derived from TD3
#   (https://github.com/sfujim/TD3)
# Copyright (c) 2020 Scott Fujimoto, licensed under the MIT License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import math
import numpy as np
import torch


class ReplayBufferVIN(object):
	def __init__(self, state_dim, action_dim, im_size, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.image = np.zeros((max_size, 2, im_size, im_size))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cpu")

	def add(self, state, action, next_state, reward, done, image):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.image[self.ptr, :, :, :] = image.cpu()

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.LongTensor(self.state[ind]),# .to(self.device),
			torch.LongTensor(self.action[ind]),# .to(self.device),
			torch.LongTensor(self.next_state[ind]),# .to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.LongTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.image[ind]).to(self.device)
		)


class ReplayBufferVIN3D(object):
	def __init__(self, state_dim, action_dim, im_size, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.image = np.zeros((max_size, 2, im_size, 12, 8))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cpu")

	def add(self, state, action, next_state, reward, done, image):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.image[self.ptr, :, :, :, :] = image.cpu()

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.LongTensor(self.state[ind]),# .to(self.device),
			torch.LongTensor(self.action[ind]),# .to(self.device),
			torch.LongTensor(self.next_state[ind]),# .to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.LongTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.image[ind]).to(self.device)
		)


class ReplayBufferVIN3(object):
	def __init__(self, state_dim, action_dim, im_size, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.image = np.zeros((max_size, 3, im_size, im_size))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cpu")

	def add(self, state, action, next_state, reward, done, image):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.image[self.ptr, :, :, :] = image.cpu()

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.LongTensor(self.state[ind]),# .to(self.device),
			torch.LongTensor(self.action[ind]),# .to(self.device),
			torch.LongTensor(self.next_state[ind]),# .to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.LongTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.image[ind]).to(self.device)
		)


class ReplayBufferHIROCNN(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, 3, state_dim, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, 3, state_dim, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.state_hl = np.zeros((max_size, 12))
		self.next_state_hl = np.zeros((max_size, 12))

		self.state_seq = list(np.zeros(max_size))
		self.action_seq = list(np.zeros(max_size))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cpu")

	def add(self, state, action, next_state, reward, done, state_seq, action_seq, state_hl, next_state_hl):
		self.state[self.ptr, :, :, :] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr, :, :, :] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.state_seq[self.ptr] = state_seq
		self.action_seq[self.ptr] = action_seq

		self.state_hl[self.ptr] = state_hl
		self.next_state_hl[self.ptr] = next_state_hl

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def off_policy_correction(self, policy):
		starts = self.state_hl[:self.size, :2]
		ends = self.next_state_hl[:self.size, :2]

		goal_diff = (ends - starts)[:, np.newaxis, :]
		goal_original = self.action[:self.size, np.newaxis, :]
		goals_sampled = np.random.normal(loc=goal_diff, scale=0.5, size=(self.size, 8, 2)).clip(-2, 2)

		seq_lens = [len(list(e)) for e in self.state_seq[:self.size]]
		total_len = sum(seq_lens)
		total_idx = list(np.cumsum(seq_lens)[:-1])

		goal_cand = np.concatenate([goal_diff, goal_original, goals_sampled], axis=1)
		s_seq = np.array(self.state_seq[:self.size])

		s_seq_flat = np.concatenate(self.state_seq[:self.size])
		a_seq_flat = np.concatenate(self.action_seq[:self.size])

		log_probs = np.zeros((10, self.size))

		for kg in range(10):
			sg = np.concatenate([(gc[kg] + e[0, :2])[None, :] - e[:, :2] for gc, e in zip(goal_cand, s_seq)])
			# sg = (goal_cand[:, kg] + s_seq[:, 0, :2])[:, None] - s_seq[:, :, :2]
			sg = sg.reshape((total_len, 2))
			a_pol = policy.select_action(torch.tensor(np.concatenate((s_seq_flat[:, 2:-2], sg), axis=1), dtype=torch.float).unsqueeze(0))
			a_diff = -0.5 * np.linalg.norm(a_pol - a_seq_flat, axis=1)
			a_diff_split = np.split(a_diff, total_idx)
			log_probs[kg] = np.array([np.sum(e) for e in a_diff_split])

		a_idx = np.argmax(log_probs, axis=0)
		self.action[:self.size, :] = goal_cand[np.arange(len(goal_cand)), a_idx, :]


class ReplayBufferHIRO(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim ))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim ))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.state_seq = list(np.zeros(max_size))
		self.action_seq = list(np.zeros(max_size))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cpu")

	def add(self, state, action, next_state, reward, done, state_seq, action_seq):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.state_seq[self.ptr] = state_seq
		self.action_seq[self.ptr] = action_seq

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def off_policy_correction(self, policy):
		starts = self.state[:self.size, :2]
		ends = self.next_state[:self.size, :2]

		goal_diff = (ends - starts)[:, np.newaxis, :]
		goal_original = self.action[:self.size, np.newaxis, :]
		goals_sampled = np.random.normal(loc=goal_diff, scale=0.5, size=(self.size, 8, 2)).clip(-2, 2)

		seq_lens = [len(list(e)) for e in self.state_seq[:self.size]]
		total_len = sum(seq_lens)
		total_idx = list(np.cumsum(seq_lens)[:-1])

		goal_cand = np.concatenate([goal_diff, goal_original, goals_sampled], axis=1)
		s_seq = np.array(self.state_seq[:self.size])

		s_seq_flat = np.concatenate(self.state_seq[:self.size])
		a_seq_flat = np.concatenate(self.action_seq[:self.size])

		log_probs = np.zeros((10, self.size))

		for kg in range(10):
			sg = np.concatenate([(gc[kg] + e[0, :2])[None, :] - e[:, :2] for gc, e in zip(goal_cand, s_seq)])
			# sg = (goal_cand[:, kg] + s_seq[:, 0, :2])[:, None] - s_seq[:, :, :2]
			sg = sg.reshape((total_len, 2))
			a_pol = policy.select_action(torch.tensor(np.concatenate((s_seq_flat[:, :-2], sg), axis=1), dtype=torch.float).unsqueeze(0))
			a_diff = -0.5 * np.linalg.norm(a_pol - a_seq_flat, axis=1)
			a_diff_split = np.split(a_diff, total_idx)
			log_probs[kg] = np.array([np.sum(e) for e in a_diff_split])

		a_idx = np.argmax(log_probs, axis=0)
		self.action[:self.size, :] = goal_cand[np.arange(len(goal_cand)), a_idx, :]


class ReplayBufferHIROParkingAngleNew(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.state_seq = list(np.zeros(max_size))
		self.action_seq = list(np.zeros(max_size))

		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = torch.device("cpu")

	def add(self, state, action, next_state, reward, done, state_seq, action_seq):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.state_seq[self.ptr] = state_seq
		self.action_seq[self.ptr] = action_seq

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def off_policy_correction(self, policy):
		starts = self.state[:self.size, [0, 1, 4, 5]]
		ends = self.next_state[:self.size, [0, 1, 4, 5]]

		start_angles = np.arctan2(starts[:self.size, 3], starts[:self.size, 2])
		end_angles = np.arctan2(ends[:self.size, 3], ends[:self.size, 2])
		diff_angles = (end_angles - start_angles)[:, np.newaxis]
		diff_positions = ends[:, :2] - starts[:, :2]

		goal_diff = np.concatenate((diff_positions, diff_angles), 1)[:, np.newaxis, :]
		goal_original = self.action[:self.size, np.newaxis, :]
		goals_sampled = np.random.normal(loc=goal_diff, scale=(0.01, 0.01, math.pi/4), size=(self.size, 8, 3)).clip((-0.04, 0.04, -math.pi), (0.04, 0.04, math.pi))

		seq_lens = [len(list(e)) for e in self.state_seq[:self.size]]
		total_len = sum(seq_lens)
		total_idx = list(np.cumsum(seq_lens)[:-1])

		goal_cand_angle = np.concatenate((goal_diff, goal_original, goals_sampled), axis=1)

		s_seq = np.array(self.state_seq[:self.size])

		s_seq_flat = np.concatenate(self.state_seq[:self.size])
		a_seq_flat = np.concatenate(self.action_seq[:self.size])

		log_probs = np.zeros((10, self.size))

		for kg in range(10):
			sg = np.concatenate([(gc[kg] + e[0, [0, 1, 4]])[None, :] - e[:, [0, 1, 4]] for gc, e in zip(goal_cand_angle, s_seq)])
			sg = sg.reshape((total_len, 3))
			sg_cos = np.cos(sg[:, 2])
			sg_sin = np.sin(sg[:, 2])
			a_pol = policy.select_action(torch.tensor(np.concatenate((sg[:, :2], s_seq_flat[:, 2:4], sg_cos[:, np.newaxis], sg_sin[:, np.newaxis]), axis=1), dtype=torch.float).unsqueeze(0))
			a_diff = -0.5 * np.linalg.norm(a_pol - a_seq_flat, axis=1)
			a_diff_split = np.split(a_diff, total_idx)
			log_probs[kg] = np.array([np.sum(e) for e in a_diff_split])

		a_idx = np.argmax(log_probs, axis=0)
		self.action[:self.size, :] = goal_cand_angle[np.arange(len(goal_cand_angle)), a_idx, :]
