# SW-MAIN/deeplearning/evaluation/evaluate.py

import sys
import os
# SW-MAIN 폴더를 모듈 검색 경로에 추가 (모듈 찾기 문제 해결)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from deeplearning.models.model import CNNAE_LSTM_Transformer
from deeplearning.dataset_loader.falling_dataset import FallingDataset

def evaluate_model(model, dataloader, device):
    """
    모델을 평가하고, 라벨, 예측값, 그리고 양성 클래스의 확률값(probability)을 반환합니다.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []  # 양성 클래스(낙상) 예측 확률

    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, _, _ = model(inputs)
            probs = softmax(logits)   # 확률 계산
            _, preds = torch.max(probs, 1)
            
            # 양성 클래스의 확률 값은 probs[:, 1] (낙상 클래스가 1이라고 가정)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def main():
    # 평가용 transform
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 테스트 세트 준비 (test_list.txt 파일 사용)
    test_dataset = FallingDataset(
        root_dir="./dataset",  # 실제 데이터 경로에 맞게 수정 (예: "./data/processed"로 변경 가능)
        list_file="./deeplearning/data_preprocessing/test_list.txt",
        transform=transform_pipeline,
        seq_length=30
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 저장된 모델 불러오기 (학습 완료된 모델)
    model = CNNAE_LSTM_Transformer(
        ae_latent_dim=256,
        lstm_hidden_dim=256,
        lstm_num_layers=2,  # 학습할 때와 동일하게
        transformer_d_model=256,
        transformer_nhead=4,
        transformer_num_layers=1,
        num_classes=2
    ).to(device)
    model.load_state_dict(torch.load("cnn_ae_lstm_transformer.pth", map_location=device))
    
    # 평가 수행: 실제 라벨, 예측값, 양성 클래스 확률 값 반환
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    
    # 평가 지표 출력
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Falling", "Falling"]))
    
    # 혼동행렬 시각화
    plot_confusion_matrix(y_true, y_pred, class_names=["Not Falling", "Falling"])
    
    # ROC Curve 시각화
    plot_roc_curve(y_true, y_probs)

if __name__ == "__main__":
    main()
