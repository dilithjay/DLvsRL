import io
import os
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

import cv2
from serial import Serial
import time
from scipy import stats

vc = cv2.VideoCapture(0) # or cap = cv2.VideoCapture("<video-path>")
# ser = Serial('COM5', 9600)

def run_inference(model, cap):
    t = time.time()
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
        line_thickness=5)
    index = np.argmax(output_dict['detection_scores'])
    # print('inference took:', time.time() - t, "seconds")
    if output_dict['detection_scores'][index] > 0.9:
        box = output_dict['detection_boxes'][index]
        center = (box[1] + box[3])/2
        return center, (box[0] + box[2])/2, img
    else:
        return None, None, None


use_model = 'u' in input("Use model (u) or collect data (d): ").lower()
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[3]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_1.load_weights('model.h5')
while vc.isOpened():
    print("=========================================")
    centers = []
    count = 0
    centers = []
    h = 0
    count = 0
    while h < 0.8 or len(centers) < 5:
        center, h, img = run_inference(model, vc)
        print("center:",center, "h:", h)
        if center:
            centers.append(center)
        else:
            h = 0
            count += 1
            if count > 2 and len(centers) != 0:
                count = 0
                print("Resetting state")
                centers = []
    if input("Skip? ") == 'y':
        continue
    n = len(centers)
    if use_model:
        mid = n // 2
        center = model_1.predict([[centers[mid - 2], centers[mid], centers[mid + 2]]])[0][0]
    else:
        for i in range(n - 4):
            data = " ".join(list(map(str, [centers[i], centers[i + 2], centers[i + 4], centers[-1]])))
            print("Writing data to file")
            with open("data.txt", "a+") as f:
                f.write(data + "\n")
        center = centers[-1]
    print("center", center)
    cur_servo_pos = (1 - center) * 180
    print("Servo Rotation:", cur_servo_pos)
    
    """
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break"""
