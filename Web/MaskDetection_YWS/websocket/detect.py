from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def mask_detector(img):
    facenet = cv2.dnn.readNet('./models/deploy.prototxt', './models/res10_300x300_ssd_iter_140000.caffemodel')
    model = load_model('./models/mask_detector')
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300,  300), mean=(104.,  177., 123.))
    facenet.setInput(blob)
    detections = facenet.forward()

    confidence = detections[0, 0, 0, 2]
    if confidence < 0.5:
        x1 = int(detections[0, 0, 0, 3] * w)
        y1 = int(detections[0, 0, 0, 4] * h)
        x2 = int(detections[0, 0, 0, 5] * w)
        y2 = int(detections[0, 0, 0, 6] * h)

        face = img[y1:y2, x1:x2]

        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = face_input[np.newaxis, ...]

        nomask = model.predict(face_input).squeeze()

        if nomask > 0.5:
            color = (0, 0, 255)
            label = "NoMask {:.2f}".format(nomask * 100)
        else:
            color = (0, 255, 0)
            label = "Mask {:.2f}".format((1-nomask) * 100)
        
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

        return img