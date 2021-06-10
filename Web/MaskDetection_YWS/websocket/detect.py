from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

import cvlib as cv
import numpy as np
import cv2

model = load_model('./models/mask_detector.h5')

def mask_detector(img):
    # global model
    total_cnt, nomask_cnt = 0, 0
    faces, confidences = cv.detect_face(img)

    for i, face in enumerate(faces):
        if confidences[i] < 0.5:
            continue
        
        x1, y1 = face[:2]
        x2, y2 = face[2:]

        face = img[y1:y2, x1:x2]

        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = face_input[np.newaxis, ...]

        mask, nomask = model.predict(face_input).squeeze()

        if mask > nomask:
            color = (0, 255, 0)
            label = "Mask {:.2f}".format(mask * 100)
        else:
            color = (0, 0, 255)
            label = "No Mask {:.2f}".format(nomask * 100)
            nomask_cnt += 1

        total_cnt += 1
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)
        
    result = (nomask_cnt / total_cnt) * 100
    # print(result)
            
    if total_cnt != 0:
        result = int(nomask_cnt/total_cnt) * 100
        result = str(result).zfill(3)
    else:
        result = '000'
    
    return img, result