import tensorflow as tf
import numpy as np
from Blackjack import Blackjack

# Define a simplified Deep Q-Network (DQN) model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),  # State space for Blackjack environment
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')  # Two output units (hit or stand)
])

# Define the optimizer and the loss function for training the DQN
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

# Define the agent's exploration strategy (e.g., epsilon-greedy)
initial_epsilon = 1.0
epsilon_decay = 0.99  # Adjust this value to control the rate of exploration decay

# Hyperparameters
num_episodes = 1000
max_steps_per_episode = 100
games_per_episode = 10
gamma = 0.99  # Discount factor

# Training loop
for episode in range(num_episodes):
    total_episode_reward = 0
    epsilon = initial_epsilon * (epsilon_decay ** episode)

    for game in range(games_per_episode):
        bj = Blackjack()
        state = np.array([bj.player.total(), bj.dealer.value(bj.dealer.hand[0]), int('A' in bj.player.hand), int(bj.player.has_pair()), bj.turn_number])

        for step in range(max_steps_per_episode):
            if np.random.rand() < epsilon:
                action = np.random.choice(2)
            else:
                q_values = model.predict(state[np.newaxis])
                action = np.argmax(q_values[0])

            if action == 0:
                bj.hit()
            else:
                bj.stand()

            next_state = np.array([bj.player.total(), bj.dealer.value(bj.dealer.hand[0]), 0, 0, bj.turn_number])

            # Define the reward based on Blackjack outcomes
            if bj.state == bj.State.WIN:
                reward = 10
            elif bj.state == bj.State.LOSE:
                reward = -10
            else:
                reward = 2

            q_values = model.predict(state[np.newaxis])
            next_q_values = model.predict(next_state[np.newaxis])
            q_values[0, action] = reward + gamma * np.max(next_q_values[0])

            with tf.GradientTape() as tape:
                q_pred = model(state[np.newaxis], training=True)
                loss = loss_fn(q_values, q_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            state = next_state
            total_episode_reward += reward

            if not bj.state == bj.State.PLAYING:
                break

    print(f"Episode {episode + 1}: Total Reward = {total_episode_reward / games_per_episode}")

# Save the trained model
model.save("model\\blackjack_dqn_model")
