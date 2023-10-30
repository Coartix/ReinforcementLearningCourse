from tensorflow import keras
from tensorflow.keras import layers


class DQN:
    def __init__(self, state_shape, num_actions):
        """
        :param state_shape: shape of the state
        :param num_actions: number of actions
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.model = self.build_network()

    def build_network(self):
        model = keras.models.Sequential()

        model.add(layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=self.state_shape))
        model.add(layers.Conv2D(64, 4, strides=2, activation="relu"))
        model.add(layers.Conv2D(64, 3, strides=1, activation="relu"))

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(self.num_actions, activation="linear"))

        return model
