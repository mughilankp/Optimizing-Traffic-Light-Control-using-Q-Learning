from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary

import os
import sys
import optparse
import random
import traci
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from collections import deque


class TrafficAgent:
    def __init__(self):
        self.discount = 0.95
        self.explore_rate = 0.1
        self.lr = 0.0002
        self.memory = deque(maxlen=200)
        self.actions = 2
        self.model = self.create_model()

    def create_model(self):
        input1 = Input(shape=(12, 12, 1))
        conv1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input1)
        conv1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(conv1)
        flat1 = Flatten()(conv1)

        input2 = Input(shape=(12, 12, 1))
        conv2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input2)
        conv2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(conv2)
        flat2 = Flatten()(conv2)

        input3 = Input(shape=(2, 1))
        flat3 = Flatten()(input3)

        merged = keras.layers.concatenate([flat1, flat2, flat3])
        dense1 = Dense(128, activation='relu')(merged)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(self.actions, activation='linear')(dense2)

        model = Model(inputs=[input1, input2, input3], outputs=[output])
        model.compile(optimizer=RMSprop(learning_rate=self.lr), loss='mse')

        return model

    def save_data(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def pick_action(self, state):
        if np.random.rand() <= self.explore_rate:
            return random.randrange(self.actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, size):
        batch = random.sample(self.memory, size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.discount * np.amax(self.model.predict(next_state)[0])
            target_values = self.model.predict(state)
            target_values[0][action] = target
            self.model.fit(state, target_values, epochs=1, verbose=0)

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)


class TrafficManager:
    def __init__(self):
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "tools"))
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", ""), "tools"))
        except ImportError:
            sys.exit("Set SUMO_HOME environment variable properly")

    def setup_routes(self):
        random.seed(42)
        total_steps = 3600
        chances = {'horizontal': 1. / 7, 'vertical': 1. / 11, 'left': 1. / 25, 'right': 1. / 30}

        with open("traffic_routes.xml", "w") as routes:
            print('<routes>', file=routes)
            car_id = 0
            for step in range(total_steps):
                for direction, chance in chances.items():
                    if random.uniform(0, 1) < chance:
                        print(f'<vehicle id="{direction}_{car_id}" type="DEFAULT" route="{direction}" depart="{step}" />', file=routes)
                        car_id += 1
            print('</routes>', file=routes)

    def get_user_options(self):
        parser = optparse.OptionParser()
        parser.add_option("--nogui", action="store_true", default=False, help="Run without GUI")
        options, _ = parser.parse_args()
        return options


if __name__ == '__main__':
    manager = TrafficManager()
    options = manager.get_user_options()

    sumo_binary = checkBinary('sumo-gui' if not options.nogui else 'sumo')
    manager.setup_routes()

    episodes = 2000
    size = 32
    agent = TrafficAgent()

    try:
        agent.load_model('traffic_model.h5')
    except FileNotFoundError:
        print("No saved model found. Starting fresh.")

    for episode in range(episodes):
        pass  # Replace this with your simulation logic
