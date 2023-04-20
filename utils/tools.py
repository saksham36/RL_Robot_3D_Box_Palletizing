#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 04/18/2023
Scripts needed for data handling, backing up, argument parsing
'''
import os
import yaml
import argparse
from torch import save as torch_save
from shutil import copyfile, copytree


def read_config():
    '''Reads config.yaml file and returns a dictionary of the contents'''
    with open("config.yaml", "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)


def get_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--experiment-name', type=str,
                        default='Debug', help='Name of the experiment')
    # Basic arguments
    parser.add_argument('--device', type=str, default='mps',
                        help='What to use for compute [GPU, CPU,MPS]  will be called')
    parser.add_argument('--seed',   type=int, default=4, help='Random seed')
    parser.add_argument('--num-processes', type=int, default=64,
                        help='The number of parallel processes used for training')
    # Env arguments
    parser.add_argument('--sequence_length', type=int, nargs='+',
                        default=25, help='The length of sequence of blocks to be loaded')
    parser.add_argument('--observable_length', type=int, nargs='+',
                        default=5, help='The length of observable sequence of blocks')
    # Hyperparameters for A2C
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--learning-rate', type=float, default=1e-6,
                        metavar='η', help='Learning rate, only works for A2C')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')

    # Arguments for eval, logging, and printing
    parser.add_argument('--load-model', action='store_true',
                        help='Load Model')
    parser.add_argument('--training_episodes', type=int, nargs='+',
                        default=1000, help='The number of episodes to train')
    parser.add_argument('--replay_size', type=int, default=1000000,
                        help='The size of replay memory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The batch size for training')
    parser.add_argument('--updates_per_step', type=int,
                        default=1, help='The number of updates per step')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')

    parser.add_argument('--evaluate', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    # parser.add_argument('--evaluation-episodes', type=int, default=100,
    #                     metavar='N', help='Number of episodes evaluated')
    parser.add_argument('--model-path', type=str,
                        help='The path to load model')
    parser.add_argument('--model-save-interval',  type=int,
                        default=200, help='How often to save the model')
    parser.add_argument('--model-save-path', type=str,
                        default='./logs/experiment', help='The path to save the trained model')
    parser.add_argument('--print-log-interval',     type=int,
                        default=10, help='How often to print training logs')

    # Dataset arguments
    parser.add_argument('--load-dataset', action='store_true',
                        help='Load an existing dataset, otherwise the data is generated on the fly')
    parser.add_argument('--dataset-path', type=str,
                        help='The path to load dataset')

    args = parser.parse_args()

    config = read_config()
    palette_size = config['palette']
    box_size_set = []
    for box in config['boxes']:
        box_size_set.append(box)
    if not all([box[2] == box_size_set[0][2] for box in box_size_set]):
        assert 0, 'Box height needs to be same for each box'

    args.box_size_set = box_size_set
    args.palette_size = palette_size
    args.fixed_sequence = config['fixed_sequence']
    return args


def backup_files(time_string, args, policy=None):
    if args.evaluate:
        target_directory = os.path.join('./logs/evaluation', time_string)
    else:
        target_directory = os.path.join('./logs/experiment', time_string)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    copyfile('config.yaml',    os.path.join(target_directory, 'config.yaml'))
    copyfile('env.py',    os.path.join(target_directory, 'env.py'))
    copyfile('main.py',    os.path.join(target_directory, 'main.py'))
    copyfile('model.py',   os.path.join(target_directory, 'model.py'))
    copyfile('sac.py',     os.path.join(target_directory, 'sac.py'))
    copyfile('replay_memory.py', os.path.join(
        target_directory, 'replay_memory.py'))

    environment_path = './palette_envs'
    environment_name = 'PaletteDiscrete-v0'
    environment_path = os.path.join(environment_path, environment_name)
    try:
        copytree(environment_path, os.path.join(
            target_directory, environment_name))
    except:
        pass
    try:
        torch_save(policy.state_dict(), os.path.join(
            args.model_save_path, + time_string + ".pt"))
    except:
        pass
