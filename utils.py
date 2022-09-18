# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image
from six import BytesIO
import numpy as np

from paddleocr import PaddleOCR,draw_ocr



def region_ocr(region_image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en') 
    result = ocr.ocr(region_image, cls=True)
    print(result)
    if len(result) >0:
      return result[0][1][0],result[0][1][1]
    else:
      return 0 , 0

    
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