#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 04/18/2023
Scripts needed for environment
'''

import torch
import copy
import numpy as np
import torch.nn.functional as F

# from gym.spaces.box import Box
from utils.tools import read_config
from itertools import permutations, product


class Environment(object):
    def __init__(self, args, fixed_sequence=None):
        # Setting seed
        if args.seed is None:
            self.seed = np.random.randint(0, round(9e9))
        else:
            self.seed = args.seed
        np.random.seed(self.seed)

        self.args = args
        if len(args.fixed_sequence):
            self.fixed_sequence = args.fixed_sequence
        else:
            self.fixed_sequence = None
        # Setting up palette dimensions
        self.length = args.palette_size[0]
        self.width = args.palette_size[1]
        self.height = args.palette_size[2]
        self.action_location = np.array(
            tuple(product(np.arange(self.length), np.arange(self.width))))

        # Set up initial state and action space
        self.reset()

        self.difference_kernel = np.array(
            [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])

    def create_box(self, params):
        ''' Create a box with given parameters'''
        return np.array([params[0], params[1], params[2]])

    def create_sequence(self):
        ''' Create a sequence of boxes to be loaded'''
        # sequence contains the index of box to be loaded
        assert self.fixed_sequence is not None
        if self.fixed_sequence is not None:
            sequence = self.fixed_sequence
        else:
            sequence = np.random.choice(
                len(self.args.box_size_set), self.args.sequence_length)
        box_sequence = np.array([[i, 0] for i in sequence])
        return box_sequence

    def reset(self):
        ''' Reset the environment to initial state'''
        # Setting up state
        # Box state: status of all blocks to be loaded
        self.height_state = np.zeros((self.length, self.width))
        self.sequence = self.create_sequence()
        self.state = [self.height_state, self.sequence]

        # Setting up action space
        self.action_location = np.array(
            tuple(product(np.arange(self.length), np.arange(self.width))))
        self.action_orientation = dict()
        for idx, dims in enumerate(self.args.box_size_set):
            self.action_orientation[idx] = dict()
            for i, perm in enumerate(permutations(dims)):
                self.action_orientation[idx][i] = tuple(perm)

        self.action_space = (self.action_location, self.action_orientation)

        # Setting up current observation
        self.observation = self.get_observations(self.state, 0)

        return self.state, self.observation

    def get_observations(self, state, time_step):
        ''' Get observations from state'''
        return state[1][time_step: min(
            time_step+self.args.observable_length, len(state[1]))]

    def check_successful(self, state, location, orientation):
        ''' Check if the block of dims: orientation can be placed at location with current state'''
        height_state = state[0]
        length, width, height = orientation
        if location[0] + length > self.length or location[1] + width > self.width:
            return False
        # Check that the height of the block is less than the height of the palette for all spaces occupied by the block
        if np.any(height_state[location[0]: location[0]+length, location[1]: location[1]+width] + height > self.height):
            return False
        return True

    def update_state(self, state, time_step, placed_successful):
        ''' Update state after placing a block'''
        if placed_successful:
            state[1][time_step][1] = 1
        else:
            state[1][time_step][1] = -1
        return state

    def place_block(self, state, location, orientation):
        ''' Place a block at location with orientation'''
        height_state = state[0]
        length, width, height = orientation
        height_state[location[0]: location[0]+length,
                     location[1]: location[1]+width] += height
        return state  # TODO: check if state is updated

    def reward(self, state, action, orientation, placed_successful):
        ''' Calculate reward'''
        r = 0
        # Reward for successfully placing a block
        if placed_successful:
            r += np.prod(orientation) / \
                np.prod([self.length, self.width, self.height])
            # Instability penalty
            # print("Reward: ", r)
            r += self.instability_penalty(state, action, orientation)
            # print("Instability penalty: ", r*self.instability_penalty(
            # state, action, orientation))

        return r

    def instability_penalty(self, state, action, orientation):
        ''' Instability penalty via local differencing'''
        selected_location_idx = np.argmax(action[:-6])
        selected_location = self.action_location[selected_location_idx]
        selected_orientation = np.argmax(action[-6:])
        selected_height_map = state[0][selected_location[0]: selected_location[0] + orientation[0],
                                       selected_location[1]: selected_location[1] + orientation[1]]

        theta = np.sqrt(np.mean(np.square(np.arctan2(selected_height_map[0, :] - selected_height_map[-1, :],
                                                     selected_height_map.shape[1])))) +\
            np.sqrt(np.mean(np.square(np.arctan2(selected_height_map[:, 0] - selected_height_map[:, -1],
                                      selected_height_map.shape[0]))))

        return -0.1*theta/np.pi
        # return 0.01 * F.conv2d(torch.from_numpy(state[0]), self.difference_kernel)

    def step(self, time_step, state, observation, action):
        ''' Take a step in the environment'''
        selected_location_idx = np.argmax(action[:-6])
        selected_orientation = np.argmax(action[-6:])
        selected_location = self.action_location[selected_location_idx]
        block_type = observation[0][0]
        orientation = self.action_orientation[block_type][selected_orientation]

        placed_successful = self.check_successful(
            state, selected_location, orientation)
        placed_successful_text = "successful" if placed_successful else "unsuccessful"
        if placed_successful:
            self.place_block(state, selected_location, orientation)
        # Get reward
        reward = self.reward(state, action,
                             orientation, placed_successful)
        # Update observation, state, timestep
        # print(
        #     f"Placing block at loc: {selected_location} with orientation: {orientation} " + placed_successful_text + f". Reward: {reward}")

        next_state = self.update_state(
            copy.deepcopy(state), time_step, placed_successful)
        next_observation = self.get_observations(state, time_step+1)
        return next_state, next_observation, reward


def make_env(args):
    return Environment(args)


def main():
    env = Environment()


if __name__ == "__main__":
    main()
