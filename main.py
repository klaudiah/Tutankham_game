from __future__ import division
import gym
import numpy as np
from keras.optimizers import Adam
from rl.callbacks import ModelIntervalCheckpoint
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from callbacks.keras_callbacks import SubTensorBoard, TestCallback
from networks.keras_network import Network
from agents.dqn import create_agent
from processors.keras_atari import AtariProcessor
from processors.baselines_atari import make_tutankham_env, make_tutankham_env_test
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2, A2C


###    DQN Keras-RL   ###

def create_dqn_agent(memory_capacity=500000, exploration_max=1.0, exploration_min=0.1, exploration_test=0,
                     exploration_steps=1e6, frame_shape=(84, 84), window_length=4):
    env = gym.make('Tutankham-v4')
    nb_actions = env.action_space.n
    processor = AtariProcessor(frame_shape)
    memory = SequentialMemory(limit=memory_capacity, window_length=window_length)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=exploration_max, value_min=exploration_min,
                                  value_test=exploration_test, nb_steps=exploration_steps)
    network = Network().create_model(frame_shape, window_length, nb_actions)

    return create_agent(network, processor, nb_actions, policy, memory)


def train_dqn(agent, optimizer, train_episodes, episode_len, logdir, checkpoint, verbose_flag=1):
    tb_callback = [SubTensorBoard(logdir=logdir)]
    tb_callback += [ModelIntervalCheckpoint(checkpoint, 10000)]
    agent.compile(optimizer)
    env = gym.make('Tutankham-v4')
    agent.fit(env, visualize=False, nb_steps=train_episodes, verbose=verbose_flag, nb_max_episode_steps=episode_len,
              callbacks=tb_callback)


def test_dqn(agent, num_episodes, episode_len, visualize):
    env = gym.make('Tutankham-v4')
    test_callback = TestCallback()
    agent.test(env, callbacks=[test_callback], nb_episodes=num_episodes, visualize=visualize,
               nb_max_episode_steps=episode_len, verbose=0)

    rewards_list = np.array(test_callback.rewards_list)
    keys_list = np.array(test_callback.keys_list)
    timesteps_list = np.array(test_callback.timesteps_list)
    keys_reward_list = np.array(test_callback.keys_reward_list)

    print('Reward - mean: {}'.format(np.mean(rewards_list)) + ' min: {}'.format(
        np.min(rewards_list)) + ' max: {}'.format(np.max(rewards_list)))
    print('Mean keys: {}'.format(np.mean(keys_list)))
    print('Max keys: {}'.format(np.max(keys_list)))
    print('Mean keys reward: {}'.format(np.mean(keys_reward_list)))
    print('Mean reward for creatures: {}'.format(np.mean(rewards_list) - np.mean(keys_reward_list)))
    print('Mean timesteps: {}'.format(np.mean(timesteps_list)))


def dqn(exploration_steps, epochs_train, episode_len_train, checkpoint, epochs_test, episode_len_test, test_visualize,
        memory_capacity=500000, exploration_max=1.0, exploration_min=0.1, exploration_test=0, logdir='board/DQN'):

    agent = create_dqn_agent(memory_capacity=memory_capacity, exploration_max=exploration_max,
                             exploration_min=exploration_min, exploration_test=exploration_test,
                             exploration_steps=exploration_steps)

    optimizer = Adam(lr=.00025)
    train_dqn(agent, optimizer, epochs_train, episode_len_train, logdir, checkpoint)
    test_dqn(agent, epochs_test, episode_len_test, test_visualize)


###   PPO and A2C - Baselines   ###

def create_ppo_agent(num_env=16, logdir='board/PPO'):
    env = make_tutankham_env(num_env=num_env)
    return PPO2(CnnPolicy, env, verbose=1, tensorboard_log=logdir)


def create_a2c_agent(num_env=16, logdir='board/A2C'):
    env = make_tutankham_env(num_env=num_env)
    return A2C(CnnPolicy, env, verbose=1, tensorboard_log=logdir)


def train_baselines(agent, total_timesteps, checkpoint):
    agent.learn(total_timesteps=total_timesteps)
    agent.save(checkpoint)


def baselines_test(agent, num_episodes, episode_len, visualize):
    env = make_tutankham_env_test()
    state = env.reset()
    rewards_list = []
    keys_list = []
    keys_reward_list = []
    timesteps_list = []

    for epiosode in range(num_episodes):
        episode_reward = 0
        is_done = False
        timestep = 0
        keys = 0
        keys_reward = 0

        while not is_done:
            if visualize:
                env.render()

            action, _ = agent.predict(state)
            state, rewards, dones, info = env.step(action)
            episode_reward += rewards[0]

            if rewards[0] >= 10:
                keys += 1
                keys_reward += rewards[0]

            if dones[0] == True or timestep == episode_len:
                print('Episode {}'.format(epiosode) + ' finished after {}'.format(timestep) +
                      ' with reward {}'.format(episode_reward) + ' reward from keys: {}'.format(keys_reward) +
                      ' and keys: {}'.format(keys))
                is_done = True
                rewards_list.append(episode_reward)
                keys_list.append(keys)
                timesteps_list.append(timestep)
                keys_reward_list.append(keys_reward)
            timestep += 1

    rewards_list = np.array(rewards_list)
    keys_list = np.array(keys_list)
    timesteps_list = np.array(timesteps_list)

    print('Reward - mean: {}'.format(np.mean(rewards_list)) + ' min: {}'.format(
        np.min(rewards_list)) + ' max: {}'.format(np.max(rewards_list)))
    print('Mean keys: {}'.format(np.mean(keys_list)))
    print('Max keys: {}'.format(np.max(keys_list)))
    print('Mean keys reward: {}'.format(np.mean(keys_reward_list)))
    print('Mean reward for creatures: {}'.format(np.mean(rewards_list) - np.mean(keys_reward_list)))
    print('Mean timesteps: {}'.format(np.mean(timesteps_list)))


def ppo(epochs_train, epochs_test, test_visualize, num_env=16, checkpoint='ppo_weigths', logdir='board', episode_len_test=1000):
    agent = create_ppo_agent(num_env=num_env, logdir=logdir)
    train_baselines(agent, epochs_train , checkpoint=checkpoint)
    baselines_test(agent, epochs_test , episode_len_test, visualize=test_visualize)


def a2c(epochs_train, epochs_test, test_visualize, num_env=16, checkpoint='a2c_weigths', logdir='board', episode_len_test=1000):
    agent = create_a2c_agent(num_env=num_env, logdir=logdir)
    train_baselines(agent, epochs_train, checkpoint=checkpoint)
    baselines_test(agent, epochs_test, episode_len_test, visualize=test_visualize)


def load_agent(agent_name, checkpoint, epochs_test, episode_len, visualize):
    if agent_name == 'dqn':
        agent = create_dqn_agent()
        optimizer = Adam(lr=.00025)
        agent.compile(optimizer)
        agent.load_weights(checkpoint)
        test_dqn(agent, epochs_test, episode_len, visualize)

    elif agent_name == 'a2c':
        agent = A2C.load(checkpoint)
        baselines_test(agent, epochs_test, episode_len, visualize)

    elif agent_name == 'ppo':
        agent = PPO2.load(checkpoint)
        baselines_test(agent, epochs_test, episode_len, visualize)


if __name__ == '__main__':
    load_agent('dqn', 'checkpoints/dqn/5m_2exp', 5, 1000, False)
