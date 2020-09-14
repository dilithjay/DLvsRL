#############################################################################
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
#############################################################################
import cv2
from ddpg_torch import Agent
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

agent = Agent(alpha=0.0001, beta=0.001, 
                input_dims=[4], tau=0.001,
                batch_size=16, fc1_dims=64, fc2_dims=64, 
                n_actions=1)
"""
print("loading models...")
agent.load_models()
print("loaded models")
"""
n_games = 1000
filename = 'Test_alpha_' + str(agent.alpha) + '_beta_' + \
            str(agent.beta) + '_' + str(n_games) + '_games'
figure_file = 'plots/' + filename + '.png'

best_score = -1
score_history = []

cur_pos = 0
vc = cv2.VideoCapture(0)

while vc.isOpened():
    state_0 = []
    h = 0
    count = 0
    prev_pos = cur_pos
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
    correct_action = max(min(state_0[-1] * 2 - state_0[-2], 1), 0)
    print("correct action:", correct_action)
    if input("Skip? ") == 'y':
        continue
    n = len(state_0)
    for i in range(n - 4):
        observation = np.array([state_0[i], state_0[i + 2], state_0[i + 4], prev_pos])
        score = 0
        agent.noise.reset()
        action = max(min(agent.choose_action(observation), 1), -1)
        print("action:",action)
        cur_pos = (action + 1) / 2
        print("cur_pos:", cur_pos)
        done = 1
        if abs(cur_pos - correct_action) < 0.1:
            reward = 1
        else:
            reward = -1
        observation_ = np.array([state_0[i], state_0[i + 2], state_0[i + 4], cur_pos])
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        score += reward
        observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        if avg_score > best_score:
            print("avg:", avg_score, "| best:", best_score)
            best_score = avg_score

        print('episode:', i, '| score %.1f' % score,
                '| average score %.1f' % avg_score)
    if input("Save model? ") == 'y':
        print("Saving models...")
        agent.save_models()
        print("Models saved")
        
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, score_history, figure_file)
