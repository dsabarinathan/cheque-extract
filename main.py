# -*- coding: utf-8 -*-


from numpy import newaxis
import numpy as np
import cv2
import os
import argparse
import time
from region_detection import detectionClass
from utils import load_image_into_numpy_array,region_ocr
import pytesseract
import tensorflow as tf
from tqdm import tqdm
import json


if __name__ == '__main__':
         
    parser = argparse.ArgumentParser(description='cheque-extract')
    parser.add_argument("--testImagePath", type=str,dest="test_path" ,help="Path of test Images",default='./test/',action="store")
    args = parser.parse_args()
    
    regionapi = detectionClass()

    labels =['Pay','Data','Amount','Rupees','AC','Signature','MICR']
    
    output_path = './output_file/'
    if not os.path.exists(output_path):
       os.makedirs(output_path)
    
    testImagePath = args.test_path
    
    fileName = os.listdir(testImagePath)
    
    for i in tqdm(range(len(fileName))):
        recognized = {}
        start_time = time.time()
        imr = load_image_into_numpy_array(testImagePath+fileName[i])
        boxes,classes,scores = regionapi.run(imr)
        
        for row in range(len(classes)):        
    
          width = boxes[row][2]- boxes[row][0]
          height = boxes[row][3]- boxes[row][1]
          
          if scores[row]>0.12:
              
              croppedImage = imr[boxes[row][1]:boxes[row][3],boxes[row][0]:boxes[row][2]]
              sub_row ={}
              if 'MICR'==labels[classes[row]-1]:
                  text = pytesseract.image_to_string(croppedImage,lang='mcr')     
                  sub_row['ocr'] = text     
                  sub_row['pos'] =  boxes[row]  
                  recognized['MICR'] = sub_row     
              elif 'Signature' !=labels[classes[row]-1]:
                  text,conf = region_ocr(croppedImage)
                  sub_row['ocr'] =text
                  sub_row['pos'] = boxes[row]
                  sub_row['confidence'] = conf
                  recognized[labels[classes[row]-1]] = sub_row
                  
        end_time = time.time()
    
        print('predicted time', end_time-start_time)

        
        filename =output_path+"/"+fileName[0:-3]+'.json'
        with open(filename, 'w') as f:
            f.write(json.dumps(recognized , ensure_ascii=False, indent=2, separators=(',', ': ')))  

       
    print("output files saved in "+output_path)
