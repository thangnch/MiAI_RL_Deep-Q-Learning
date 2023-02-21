import random
import numpy as np
import gym
import tensorflow
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Xây dựng một class Agent
class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Khởi tạo replay buffer
        self.replay_buffer = deque(maxlen=50000)

        # Khởi tạo tham số của Agent
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10

        self.main_network = self.get_nn()
        self.target_network = self.get_nn()

        # Update weight của mạng target = mạng main
        self.target_network.set_weights(self.main_network.get_weights())

    def get_nn(self):
        model  = Sequential()
        model.add (Dense(32, activation='relu', input_dim=self.state_size))
        model.add (Dense(32, activation='relu'))
        model.add (Dense(self.action_size))
        model.compile( loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):

        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch  = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = [batch[4] for batch in exp_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def train_main_network(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        # Lấy Q value của state hiện tại
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Lấy Max Q values của state S' (State chuyển từ S với action A)
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)

        for i in range(batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
            q_values[i][action_batch[i]] = new_q_values

        self.main_network.fit(state_batch, q_values, verbose=0)

    def make_decision(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)

        state = state.reshape((1, self.state_size))
        q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

# Chương trình chính

# Khởi tạo môi trường
env = gym.make("CartPole-v1")
state, _ = env.reset()

# Định nghĩa state_size và action_size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Định nghĩa tham số khác
n_episodes = 100
n_timesteps = 500
batch_size = 64

# Khởi tạo agent
my_agent = MyAgent(state_size, action_size)
total_time_step = 0

for ep in range(n_episodes):

    ep_rewards = 0
    state, _ = env.reset()

    for t in range(n_timesteps):

        total_time_step += 1
        # Cập nhật lại target NN mỗi my_agent.update_targetnn_rate
        if total_time_step % my_agent.update_targetnn_rate == 0:
            # Có thể chọn cách khác = weight của targetnetwork = 0 * weight của targetnetwork  + 1  * weight của mainnetwork
            my_agent.target_network.set_weights(my_agent.main_network.get_weights())

        action = my_agent.make_decision(state)
        next_state, reward, terminal, _, _ = env.step(action)
        my_agent.save_experience(state,action, reward, next_state, terminal)

        state = next_state
        ep_rewards += reward

        if terminal:
            print("Ep ", ep+1, " reach terminal with reward = ", ep_rewards)
            break

        if len(my_agent.replay_buffer) > batch_size:
            my_agent.train_main_network(batch_size)

    if my_agent.epsilon > my_agent.epsilon_min:
        my_agent.epsilon = my_agent.epsilon * my_agent.epsilon_decay

# Save weights
my_agent.main_network.save("train_agent.h5")







