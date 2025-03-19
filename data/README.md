### 1. Kaggle

https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset

# raw 버전

train: 374 / val: 111 == 총: 485
[8:2:0]

라벨링
-> label 구분: Fall Detected, Walking, Sitting (클래스 3개)
Violet – Fall Detected
Blue – Walking
Green – Sitting

# --------------------------------------------------

### 2. RoboFlow -1

https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8

YOLOv8 Oriented Bounding Boxes 버전 데이터셋 다운로드함
-> 일반 YOLO 라벨과 달리, ‘Oriented’ 정보가 들어가면 회전된 직사각형/경계박스까지 표현할 수 있는 포맷을 의미

# raw 버전

train: 3148 / val: 899 / test: 450 == 총: 4497
[7:2:1]

데이터 전처리(Preprocessing)
-> Auto-Orient 적용됨
-> 카메라/디바이스마다 이미지의 방향(가로/세로 회전)이 EXIF 메타데이터로 저장되어 있는데,이를 제거하고 실제 픽셀 데이터를 화면에 맞추어 정렬시킨다는 의미

데이터 증강 (Augmentations)
-> X

라벨링
-> fall-detected 여부 (클래스 1개)

# augmentation 버전

train: 9444 / val: 899 / test: 450 == 총: 10793
[8:1:1]

데이터 전처리(Preprocessing)
-> Auto-Orient 적용됨
-> Resize to 640x640 (Stretch)

데이터 증강 (Augmentations) ->
Outputs per training example: 3
Flip: Horizontal
Crop: 0% Minimum Zoom, 20% Maximum Zoom
Rotation: Between -12° and +12°
Shear: ±2° Horizontal, ±2° Vertical
Grayscale: Apply to 10% of images
Hue: Between -20° and +20°
Saturation: Between -20% and +20%
Brightness: Between -20% and +20%
Exposure: Between -20% and +20%
Gaussian Blur: Up to 0.75px
Cutout: 5 boxes with 3% size each

# --------------------------------------------------

### 3. RoboFlow -2: Ur Fall Detection Dataset

https://universe.roboflow.com/ufddfdd/ur-fall-detection-dataset/dataset/1

2번과 이하 동일

# raw 버전

train: 2000 / val: 400 / test: 200 == 총: 2600
[7:2:1]

데이터 전처리(Preprocessing)
-> Auto-Orient 적용됨
-> Resize to 640x640 (Stretch)

데이터 증강 (Augmentations)
-> X

라벨링
-> person / fall 여부 (2개 클래스)
-> 넘어진 상태를 fall로 태깅하고, 일반 서 있거나 앉아 있는 사람을 person으로 별도 라벨링 가능

# --------------------------------------------------

### 4. RoboFlow -3: Fall detection self

https://universe.roboflow.com/fall-detection-vbh5q/fall-detection-self/dataset/1

2번과 이하 동일

# raw 버전

train: 363 / val: 111 / test: 0 == 총: 474
[::0]

데이터 전처리(Preprocessing)
-> Auto-Orient 적용됨
-> Resize to 640x640 (Stretch)

데이터 증강 (Augmentations)
-> X

라벨링
-> 0:쓰러짐 / 1: 앉음 / 2: 서있음 (3개 클래스)

# --------------------------------------------------
