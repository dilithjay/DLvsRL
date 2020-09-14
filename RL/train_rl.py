import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from serial import Serial
import time
import cv2
import keyboard

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

num_actions = 3

ser = Serial('COM5', 9600)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(80, 96, 1))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(64, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.MaxPooling2D(2, 2)(layer1)
    layer3 = layers.Conv2D(128, 3, strides=1, activation="relu")(layer2)
    #layer4 = layers.MaxPooling2D(2, 2)(layer3)
    layer5 = layers.Conv2D(128, 3, strides=1, activation="relu")(layer3)
    #layer6 = layers.MaxPooling2D(2, 2)(layer5)
    
    layer7 = layers.Flatten()(layer5)

    layer8 = layers.Dense(512, activation="relu")(layer7)
    action = layers.Dense(num_actions, activation="linear")(layer8)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()


# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.0025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 10     #50000
# Number of frames for exploration
epsilon_greedy_frames = 500    #1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

vc = cv2.VideoCapture(0)
"""
try:
    print("Loading model...")
    model.load_weights('model_rl.h5')
except:
    print('Weights not found')
"""

while True:  # Run until solved
    ###########################################################################
    """if input('Stop training?(y/n) ') == 'y':
        if input('Save model?(y/n) ') == 'y':
            model.save_weights("model_rl.h5")
        break"""
        
    # reset motor position and wait 2 secs for motion to complete
    ser.write('0'.encode())
    time.sleep(2)
    
    # h=480, w=640
    ret, img1 = vc.read()
    ret, img2 = vc.read()
    img1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)[240:, 120:520], (80, 48))/255
    img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)[240:, 120:520], (80, 48))/255
    state = np.concatenate((img1, img2)).reshape((1, 80, 96, 1))
    ###########################################################################
    
    # state = np.array(env.reset())
    episode_reward = 0
    cur_servo_pos = 90
    
    nxt = ''
    
    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
            print("Random action:", action)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = state.reshape((80, 96, 1))
            state_tensor = tf.convert_to_tensor(state_tensor)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            print("Predicted action:", action)

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        
        ###########################################################################
        angles = [30, 90, 150]
        cur_servo_pos = angles[action]
        ser.write(str(cur_servo_pos).encode())
        print("Action taken:", cur_servo_pos)
        time.sleep(1.2)
        
        ret, img1 = vc.read()
        ret, img2 = vc.read()
        img1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)[240:, 120:520], (80, 48))/255
        img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)[240:, 120:520], (80, 48))/255
        state_next = np.concatenate((img1, img2)).reshape((1, 80, 96, 1))
        reward = 0
        done = 0
        
        if keyboard.is_pressed('z'):
            nxt = input("Caught(c), Missed(m), Caught but stop traing(cs), Missed but stop training(ms)")
            if nxt[-1] == 'c':
                print('Caught ball')
                reward = 1
                done = 1
            elif nxt[-1] == 'm':
                print('Missed ball')
                reward = -1
                done = 1
        ###########################################################################
        
        # state_next, reward, done, _ = env.step(action)
        # state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            print('Predicting from target model')
            future_rewards = model_target.predict(state_next_sample.reshape((batch_size, 80, 96)))
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample.reshape((batch_size, 80, 96)))

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break
    print("epsilon:", epsilon)
    if len(nxt) > 1 and nxt[-2] == 's':
        if input('Save model?(y/n) ') == 'y':
            print("Saving model...")
            model.save_weights("model_rl.h5")
        break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 0.8 and episode_count > 5:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
