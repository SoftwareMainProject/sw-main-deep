# test_cnn.py
import torch
import torchvision.models as models
import torch.nn as nn

def test_cnn():
    # 사전 학습된 ResNet50 사용
    resnet = models.resnet50(pretrained=True)
    # 마지막 fully connected 레이어 제거 → 특징 추출기 역할
    cnn = nn.Sequential(*list(resnet.children())[:-1])
    cnn.eval()
    
    # 더미 입력: 배치 크기 1, 채널 3, 크기 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        features = cnn(dummy_input)  # 예상 출력: [1, 512, 1, 1] (ResNet18 기준)
    
    # 출력 벡터를 평탄화: [1, 512]
    features = features.view(features.size(0), -1)
    print("CNN output features shape:", features.shape)

if __name__ == "__main__":
    test_cnn()
