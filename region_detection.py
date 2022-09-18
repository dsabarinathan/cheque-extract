# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:33:26 2022

@author: SABARI
"""

import tensorflow as tf
import numpy as np


class detectionClass:
    path = './model/saved_model'
    
    def __init__(self):
        self._model = tf.saved_model.load(self.path)
    
    def run(self, image):
      im_height, im_width,_ = image.shape
      image = np.asarray(image)
    
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis,...]
    
      # Run inference
      model_fn = self.model.signatures['serving_default']
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

    
      im_height, im_width,_ = image.shape
      boxes = output_dict['detection_boxes']
      print("boxes",boxes.shape)
      classes = output_dict['detection_classes']
      score = output_dict['detection_scores']
      boxes_list = [None for i in range(len(classes))]
      for i in range(len(classes)):
    
              boxes_list[i] = (int(boxes[i,1]*im_width),
              int(boxes[i,0] * im_height),
              int(boxes[i,3]*im_width),
              int(boxes[i,2] * im_height))
    
        
      return np.array(boxes_list),np.array(classes),np.array(score)
        