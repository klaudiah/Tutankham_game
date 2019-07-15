from rl.callbacks import Callback
from keras.callbacks import TensorBoard


class SubTensorBoard(TensorBoard):

    def __init__(self, logdir):
        super(SubTensorBoard, self).__init__(log_dir=logdir)

    def on_episode_end(self, epoch, logs):
        timesteps = logs['nb_steps']
        super(SubTensorBoard, self).on_epoch_end(timesteps, logs)


class TestCallback(Callback):

    def __init__(self, *args, **kwargs):
        super(TestCallback, self).__init__()
        self.keys_list = []
        self.keys_reward_list = []
        self.rewards_list = []
        self.timesteps_list = []
        self.keys = 0
        self.keys_reward = 0

    def on_step_end(self, step, logs):
        if logs['reward'] > 3:
            self.keys += 1
            self.keys_reward += logs['reward']

    def on_episode_begin(self, episode, logs={}):
        self.keys_reward = 0
        self.keys = 0

    def on_episode_end(self, episode, logs):
        """ Print logs at end of each episode """
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}, keys: {3}, keys reward: {4}'
        variables = [
            episode + 1,
            logs['episode_reward'],
            logs['nb_steps'],
            self.keys,
            self.keys_reward
        ]

        print(template.format(*variables))
        self.keys_list.append(self.keys)
        self.keys_reward_list.append(self.keys_reward)
        self.rewards_list.append(logs['episode_reward'])
        self.timesteps_list.append(logs['nb_steps'])
        self.keys = 0
        self.keys_reward = 0
