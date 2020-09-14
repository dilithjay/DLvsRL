import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


labelmap_path = "labelmap.txt"
output_directory = "inference_graph"

category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

tf.keras.backend.clear_session()
model = tf.saved_model.load('inference_graph/saved_model')

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict
"""
import cv2
cap = cv2.VideoCapture(0) # or cap = cv2.VideoCapture("<video-path>")
"""
def run_inference(model, cap):
    ret, image_np = cap.read()
    img = image_np[240:, 120:520, :]
    # Actual detection.
    output_dict = run_inference_for_single_image(model, img)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=.9,
        line_thickness=8)
    index = np.argmax(output_dict['detection_scores'])
    if output_dict['detection_scores'][index] > 0.9:
        box = output_dict['detection_boxes'][index]
        center = (box[1] + box[3])/2
        return center, (box[0] + box[2])/2, img
    else:
        return None, None, None

# run_inference(model, cap)


#################################################################################################


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
batch_size = 16  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

num_actions = 3

ser = Serial('COM5', 9600)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(4))
    layer1 = layers.Dense(16, activation="relu")(inputs)
    layer2 = layers.Dense(16, activation="relu")(layer1)
    action = layers.Dense(num_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model_rl = create_q_model()
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
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10
# Using huber loss for stability
loss_function = keras.losses.Huber()

vc = cv2.VideoCapture(0)

print("Loading model...")
try:
    model_rl.load_weights('model_rl.h5')
    print("Loaded model")
except:
    print('Weights not found')

epsilon = float(input("Enter starting epsilon: "))

angles = [30, 90, 150]
cur_servo_pos = 0
ser.write('0'.encode())
time.sleep(2)

counts = [0, 0, 0]

while vc.isOpened():  # Run until solved
    ###########################################################################    
    
    state_0 = []
    h = 0
    count = 0
    while h < 0.8 or len(state_0) < 5:
        center, h, img = run_inference(model, vc)
        print("center:",center, "h:", h)
        if center:
            state_0.append(center)
        else:
            h = 0
            count += 1
            if count > 2 and len(state_0) != 0:
                count = 0
                print("Resetting state")
                state_0 = []
        """cv2.imshow('object_detection', img)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        vc.release()
        cv2.destroyAllWindows()
        break"""
    
    correct_action = int(input("What was the correct action? "))
    if correct_action > 2:
        continue
    n = len(state_0)
    counts[correct_action] += n
    prev_pos = cur_servo_pos
    if epsilon > 0.1:
        print("No of frames:",n)
        for i in range(n - 4):
            state = np.array([state_0[i], state_0[i + 2], state_0[i + 4], prev_pos/180])
            print("state:", state)
            
            episode_reward = 0
            # cur_servo_pos = 90
            
            nxt = ''
            
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                state_tensor = state.reshape((4, 1))
                state_tensor = tf.convert_to_tensor(state_tensor)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model_rl(state_tensor, training=False)
                print("action probs:", action_probs)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                print("Predicted action:", action)
                # Take random action
                action = np.random.choice(num_actions)
                print("Random action:", action)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = state.reshape((4, 1))
                state_tensor = tf.convert_to_tensor(state_tensor)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model_rl(state_tensor, training=False)
                print("action probs:", action_probs)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                print("Predicted action:", action)

            # Decay probability of taking random action
            epsilon -= 0.000005     # epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
            print("epsilon:", epsilon)
            
            cur_servo_pos = angles[action]
            state_next = np.array([state_0[i], state_0[i + 2], state_0[i + 4], cur_servo_pos/180])
            
            reward = 0
            done = 0
            
            if action == correct_action:
                print('Caught ball')
                reward = 1
                done = 1
            else:
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
                print("##################################################################")
                print('Predicting from target model')
                future_rewards = model_target.predict(state_next_sample.reshape((batch_size, 4)))
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
                    q_values = model_rl(state_sample.reshape((batch_size, 4)))

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)
                    print(loss)

                # Backpropagation
                grads = tape.gradient(loss, model_rl.trainable_variables)
                optimizer.apply_gradients(zip(grads, model_rl.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model_rl.get_weights())
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
                    
    else:
        state = np.array([state_0[-1], state_0[-3], state_0[-5], prev_pos/180])
        episode_reward = 0
        # cur_servo_pos = 90
        
        nxt = ''
        frame_count += 1

        # Use epsilon-greedy for exploration
        if epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
            print("Random action:", action)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = state.reshape((4, 1))
            state_tensor = tf.convert_to_tensor(state_tensor)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model_rl(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            print("Predicted action:", action)
        
        cur_servo_pos = angles[action]
        state_next = np.array([state_0[-1], state_0[-3], state_0[-5], prev_pos/180])
        
        reward = 0
        done = 1
        
        if action == correct_action:
            print('Caught ball')
            reward = 1
        else:
            print('Missed ball')
            reward = -1
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
            future_rewards = model_target.predict(state_next_sample.reshape((batch_size, 4)))
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
                q_values = model_rl(state_sample.reshape((batch_size, 4)))

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model_rl.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_rl.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model_rl.get_weights())
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
    
    print(counts)
    if input('Save model?(y/n) ') == 'y':
        print("Saving model...")
        model_rl.save_weights("model_rl.h5")
        print("Saved model...")
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
