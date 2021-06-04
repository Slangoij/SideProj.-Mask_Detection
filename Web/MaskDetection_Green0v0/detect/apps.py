# https://sundries-in-myidea.tistory.com/95
import tensorflow as tf
from django.apps import AppConfig
import numpy as np
import cv2
# import pillow
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent

# # 서버로딩 속도 위해 잠시 주석처리!!
# class DetectConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'detect'

# class FaceConfig(AppConfig):
#     MODEL_PATH = os.path.join(BASE_DIR, 'my_model1')
#     facenet = cv2.dnn.readNet('mask-detection-master/models/deploy.prototxt',
#                             'mask-detection-master/models/res10_300x300_ssd_iter_140000.caffemodel')
#     model = load_model(MODEL_PATH) #saved_model.pb
#     name = 'facenet'
#     print('모델 읽기 성공')


# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
# WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected when loading Keras model. Please ensure that you are saving the model with model.save() or tf.keras.models.save_model(), *NOT* tf.saved_model.save(). To confirm, there should be a file named "keras_metadata.pb" in the SavedModel directory.