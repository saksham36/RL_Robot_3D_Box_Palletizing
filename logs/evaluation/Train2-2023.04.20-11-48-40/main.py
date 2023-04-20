#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 04/18/2023
Main script
'''
import os
import time
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter
from utils.tools import get_args, backup_files
from env import make_env
from replay_memory import ReplayMemory
from sac import SAC
# from decision_transformer import DecisionTransformer
np.set_printoptions(threshold=np.inf)


def main(args):
    time_string = args.experiment_name + '-' + \
        time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    if args.device == 'mps':
        device = torch.device('mps')
    elif args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Backup all py files[great SWE practice]
    backup_files(time_string, args, None)

    # Create create tensorboard logs
    log_writer_path = './logs/runs/{}'.format('PalletePacking-' + time_string)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(logdir=log_writer_path)

    # Create packing environments to collect training samples online
    env = make_env(args)
    # Create memory
    memory = ReplayMemory(args.replay_size, args.seed)
    # Create agent
    agent = SAC(len(env.state[0].flatten(
    )) + len(env.state[1].flatten()), env.action_space, args, device)
    # agent = DecisionTransformer(
    #     state_dim=len(env.state[0].flatten()) + len(env.observation.flatten()),
    #     act_dim=env.length * env.width + 6,
    #     n_blocks=args.n_blocks,
    #     h_dim=args.hidden_dim,
    #     context_len=args.context_len,
    #     n_heads=args.n_heads,
    #     drop_p=args.dropout_p,
    # ).to(device)

    # optimizer = torch.optim.AdamW(
    #     agent.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.wt_decay
    # )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lambda steps: min((steps+1)/args.warmup_steps, 1)
    # )

    # Load the trained model, if needed
    if args.load_model:
        agent.load_model(args.model_path)

    # Perform all training.
    total_numsteps = 0
    updates = 0
    for i_episode in range(args.training_episodes):
        episode_reward = 0
        episode_steps = 0
        state, observation = env.reset()

        for t in range(len(state[1])):
            # Sample action from policy
            action = agent.select_action(state, observation)
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar(
                        'entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, next_observation, reward = env.step(
                t, state, observation, action)  # Step

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Append transition to memory
            memory.push(torch.cat([torch.from_numpy(state[0].flatten()), torch.from_numpy(state[1].flatten())]), action, reward, torch.cat(
                [torch.from_numpy(next_state[0].flatten()), torch.from_numpy(next_state[1].flatten())]))
            state = next_state
            observation = next_observation

        writer.add_scalar('reward/train', episode_reward, i_episode)
        if i_episode % args.print_log_interval == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
                i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % args.model_save_interval == 0:
            agent.save_checkpoint(args.model_save_path, time_string, i_episode)

        if i_episode % 10 == 0 and args.evaluate is True:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state, observation = env.reset()
                episode_reward = 0
                for t in range(len(state[1])):
                    action = agent.select_action(state, evaluate=True)
                    next_state, next_observation, reward = env.step(
                        t, state, observation, action)
                    episode_reward += reward
                    state = next_state
                    observation = next_observation
                avg_reward += episode_reward
            avg_reward /= episodes

            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print(20*"-")
            print("Test Episodes: {}, Avg. Reward: {}".format(
                episodes, round(avg_reward, 2)))
            print(20*"-")


if __name__ == '__main__':
    args = get_args()
    main(args)
