```python
from keras.models import Model, load_model
from math import atan2, degrees
import dlib, cv2, keras, sys, os,
import pandas as pd
import numpy as np
```

## face detector & landmark predictor
- 마스크를 씌우기 위해 사람의 얼굴 인식 : face detector
- 마스크 위치와 기울기를 지정하기 위한 : landmark predictor


```python
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
```

- mask 이미지를 PNG로 4차원으로 받기 위한 cv2.IMREAD_UNCHANGED 사용

```python
mask = cv2.imread('./bluemask.png',cv2.IMREAD_UNCHANGED)
```

### Path

```python
base_path = './test_img/'
file_list = sorted(os.listdir(base_path))
```

## overlay function
- 발견한 위치에 마스크를 씌우는 함수
- [bitwise_and](https://copycoding.tistory.com/156) 사용


```python
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # BGR채널에 Alpha 채널까지!
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

    # 다시 BGR채널로!
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img
```

## angle between
- 얼굴의 각도에 따라 연산을 수행
- landmark로 얻어낸 결과 49, 55번 양쪽 입꼬리를 기준으로 계산


```python
def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))
```


```python
for f in file_list:
    img = cv2.imread(os.path.join(base_path,f))
    ori = img.copy() # original

    # detect faces
    faces = detector(img)
    
    # only one face
    if not faces or len(faces) > 1:
        continue
        
    face = faces[0]
    
    # 얼굴 특징점 추출
    dlib_shape = predictor(img, face)

    # 68개의 점 x,y를 array로 shape_2d에 저장
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    
    # shape_2d의 49, 55를 저장
    left_x, left_y = shape_2d[48]
    right_x, right_y = shape_2d[54]
    mask_x, mask_y = shape_2d[51]
    
    # compute center of face
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
    
    # 얼굴 크기만큼 resize(overlay)
    face_size = int(max(bottom_right - top_left)*1.5)
    
    # 마스크 각도
    angle = -angle_between((left_x, left_y), (right_x, right_y))
    M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1)
    rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    try:
        result = overlay_transparent(ori, rotated_mask, mask_x, mask_y, overlay_size=(face_size, face_size))
    except:
        pass
        # print('failed overlay image')
    
    # 실제 특징점 파악을 위한 데이터 확인
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(),face.bottom()),
                       color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    
    for s in shape_2d: #얼굴특징점 68개
        cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

    cv2.circle(img, center=tuple(top_left), radius=1, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((left_x, left_y)), radius=1, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((right_x, right_y)), radius=1, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.imshow('result',result)
    cv2.waitKey()    
    cv2.destroyAllWindows()
    
    # Save
    filename, ext = os.path.splitext(f)
    cv2.imwrite('./result/%s_result%s' % (filename, ext), result)
```
