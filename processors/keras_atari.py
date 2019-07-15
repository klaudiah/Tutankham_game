from rl.core import Processor
from PIL import Image
import numpy as np


class AtariProcessor(Processor):

    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        super(Processor)

    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.frame_shape).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        alpha = 0.3
        if reward <= 3:
            return reward * alpha
        else:
            return reward / alpha
