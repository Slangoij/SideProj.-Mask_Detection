import cv2
import base64
import numpy as np
from detect import mask_detector

def from_b64(uri):
    '''
        Convert from b64 uri to OpenCV image
        Sample input: 'data:image/jpg;base64,/9j/4AAQSkZJR......'
    '''
    encoded_data = uri.split(',')[1]
    data = base64.b64decode(encoded_data)
    np_arr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def to_b64(img):
    '''
        Convert from OpenCV image to b64 uri
        Sample output: 'data:image/jpg;base64,/9j/4AAQSkZJR......'
    '''
    _, buffer = cv2.imencode('.jpg', img)
    uri = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpg;base64,{uri}'

def img_processer(data):
    try:
        img = from_b64(data) #     <<<< 이미지 받기
        # Do some OpenCV Processing
        img, result = mask_detector(img) #           <<<< client에서 보낸 img에 예측한거 그리기
        # End for OpenCV Processing
        # print(to_b64(img)[:10] +  '{03d}'.format(result))
        return to_b64(img) + result #       <<<<<< 이미지 보내기
    except:
        # just in case some process is failed
        # normally, for first connection
        # return the original data
        return data # + '000'
