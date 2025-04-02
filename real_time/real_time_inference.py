# SW-MAIN/deeplearning/real_time/real_time_inference_improved.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from deeplearning.models.model import CNNAE_LSTM_Transformer

# --------------------------
# 1. DNN 기반 사람 검출 (MobileNet SSD)
# --------------------------
def init_person_detector():
    proto_path = os.path.join("deeplearning", "models", "dnn", "MobileNetSSD_deploy.prototxt")
    model_path = os.path.join("deeplearning", "models", "dnn", "MobileNetSSD_deploy.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net

def detect_person(net, frame, conf_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            idx = int(detections[0, 0, i, 1])
            # MobileNet SSD에서 'person' 클래스가 보통 인덱스 15입니다.
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))
    return boxes

# --------------------------
# 2. 전처리 함수 (학습 시와 동일)
# --------------------------
def preprocess_frame(frame, target_size=(224,224), transform_pipeline=None):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    pil_img = pil_img.resize(target_size)
    if transform_pipeline:
        return transform_pipeline(pil_img)
    else:
        return pil_img

# --------------------------
# 3. 메인 함수 (실시간 추론)
# --------------------------
def main():
    # 전처리 파이프라인: 학습 시와 동일
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 출력 임계값 및 temporal smoothing 설정
    threshold = 0.9           # Falling 예측 임계값
    buffer_window = 5         # 최근 5 시퀀스 결과로 평균 계산
    prediction_buffer = []    # temporal smoothing 버퍼

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 모델 로드
    model = CNNAE_LSTM_Transformer(
        ae_latent_dim=256,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        transformer_d_model=256,
        transformer_nhead=4,
        transformer_num_layers=1,
        num_classes=2
    ).to(device)
    model.load_state_dict(torch.load("cnn_ae_lstm_transformer.pth", map_location=device))
    model.eval()

    # DNN 기반 사람 검출 초기화 (MobileNet SSD)
    person_detector = init_person_detector()

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    seq_length = 30
    frame_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # DNN 기반 사람 검출
        boxes = detect_person(person_detector, frame, conf_threshold=0.5)
        if len(boxes) == 0:
            cv2.putText(frame, "No Person", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            frame_buffer = []
            prediction_buffer = []
        else:
            # 여러 박스 중 가장 큰 박스를 선택하여 ROI 결정
            max_area = 0
            for (startX, startY, endX, endY) in boxes:
                area = (endX - startX) * (endY - startY)
                if area > max_area:
                    max_area = area
                    best_box = (startX, startY, endX, endY)
            (startX, startY, endX, endY) = best_box
            # ROI 확장 (10% 확장)
            deltaX = int(0.1 * (endX - startX))
            deltaY = int(0.1 * (endY - startY))
            roi = frame[max(0, startY-deltaY):min(frame.shape[0], endY+deltaY),
                        max(0, startX-deltaX):min(frame.shape[1], endX+deltaX)]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
            
            # 여기서는 ROI 전체 또는 원본 frame에 대해 전처리 적용할 수 있음
            processed_frame = preprocess_frame(frame, target_size=(224,224), transform_pipeline=transform_pipeline)
            frame_buffer.append(processed_frame)
            
            if len(frame_buffer) == seq_length:
                seq_tensor = torch.stack(frame_buffer, dim=0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _, _ = model(seq_tensor)
                    prob = torch.softmax(logits, dim=1)
                    fall_prob = prob[0, 1].item()
                # 임계값 적용
                prediction_buffer.append(fall_prob)
                if len(prediction_buffer) > buffer_window:
                    prediction_buffer.pop(0)
                avg_fall_prob = np.mean(prediction_buffer)
                final_pred = 1 if avg_fall_prob >= threshold else 0
                label_text = "Falling" if final_pred == 1 else "Not Falling"
                cv2.putText(frame, f"{label_text} (avg={avg_fall_prob:.2f})", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                frame_buffer = []

        cv2.imshow("Real-time Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
