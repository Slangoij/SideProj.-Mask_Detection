# 마스크씌우기 파이프라인
```python
# Necessary imports
import cv2
import dlib
import numpy as np
import os
import imutils

# set directories
IMG_DIR_PATH = r"C:\Users\inje.jeong\Desktop\images"
os.chdir(IMG_DIR_PATH)
file_list = os.listdir(IMG_DIR_PATH)

#Initialize color [color_type] = (Blue, Green, Red)
color_blue = (239,207,137)
color_cyan = (255,200,0)
color_black = (0, 0, 0)
color_white = (255,255,255)
file_list[:5]

os.getcwd()
choice1 = color_white # 마스크 색깔
choice2 = 2 # 마스크 모양 0 ~ 2

# 이미지 있는 디렉토리로 설정
IMG_DIR_PATH = r"C:\Users\inje.jeong\Desktop\images"
# 주피터노트북 파일 있는 디렉토리로 설정
BASE_PATH = r"E:\Users\inje\교육\202012_국비지원 IT교육\SideProj.-Mask_Detection"
os.chdir(IMG_DIR_PATH)
if not os.path.isdir('output'):
    os.mkdir('output')
OUTPUT_DIR_PATH = os.path.join(IMG_DIR_PATH, 'output')
os.chdir(BASE_PATH)

# 각 이미지마다 처리
for idx, img_path in enumerate(file_list):
    IMG_PATH = os.path.join(IMG_DIR_PATH, img_path)
    img = cv2.imread(IMG_PATH)
    img = imutils.resize(img, width = 500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    
    faces = detector(gray, 1)
    
    # Initialize dlib's shape predictor
    # Get the shape using the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # 이미지 내에서 탐지된 얼굴마다 
    for face in faces:
        landmarks = predictor(gray, face)

        points = []
        for i in range(1, 16):
            point = [landmarks.part(i).x, landmarks.part(i).y]
            points.append(point)

        # Coordinates for the additional 3 points for wide, high coverage mask - in sequence
        mask_a = [((landmarks.part(42).x), (landmarks.part(15).y)),
                  ((landmarks.part(27).x), (landmarks.part(27).y)),
                  ((landmarks.part(39).x), (landmarks.part(1).y))]

        # Coordinates for the additional point for wide, medium coverage mask - in sequence
        mask_c = [((landmarks.part(29).x), (landmarks.part(29).y))]

        # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
        mask_e = [((landmarks.part(35).x), (landmarks.part(35).y)),
                  ((landmarks.part(34).x), (landmarks.part(34).y)),
                  ((landmarks.part(33).x), (landmarks.part(33).y)),
                  ((landmarks.part(32).x), (landmarks.part(32).y)),
                  ((landmarks.part(31).x), (landmarks.part(31).y))]

        fmask_a = points + mask_a
        fmask_c = points + mask_c
        fmask_e = points + mask_e

        # mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
        # mask_type[choice2]

        # Using Python OpenCV – cv2.polylines() method to draw mask outline for [mask_type]:
        # fmask_a = wide, high coverage mask,
        # fmask_c = wide, medium coverage mask,
        # fmask_e  = wide, low coverage mask

        fmask_a = np.array(fmask_a, dtype=np.int32)
        fmask_c = np.array(fmask_c, dtype=np.int32)
        fmask_e = np.array(fmask_e, dtype=np.int32)

        mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
        # mask_type[choice2]

        # change parameter [mask_type] and color_type for various combination
        img2 = cv2.polylines(img, [mask_type[choice2]], True, choice1,
                             thickness=2, lineType=cv2.LINE_8)

        # Using Python OpenCV – cv2.fillPoly() method to fill mask
        # change parameter [mask_type] and color_type for various combination
        img3 = cv2.fillPoly(img2, [mask_type[choice2]], choice1, lineType=cv2.LINE_AA)

        # cv2.imshow("image with mask outline", img2)
        # cv2.imshow("image with mask", img3)

#     cv2.imshow("image with mask outline", img_2)
#     cv2.imshow(f"image with mask_{idx}", img3)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    #Save the output file for testing
    outputNameofImage = f"output_{idx}.jpg"
    print("Saving output image to", outputNameofImage)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR_PATH, outputNameofImage)
    cv2.imwrite(OUTPUT_PATH, img3)
```
