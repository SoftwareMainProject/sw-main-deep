# SW-MAIN/deeplearning/models/model.py

import torch
import torch.nn as nn
import torchvision.models as models

class CNNAE_LSTM_Transformer(nn.Module):
    """
    통합 모델: CNN + AutoEncoder (Encoder/Decoder) + LSTM + Transformer + 분류 FC.
    최종 Loss = CrossEntropyLoss + alpha * MSELoss.
    Dropout은 LSTM과 Transformer에 적용합니다.
    """
    def __init__(self, 
                 ae_latent_dim=256,
                 lstm_hidden_dim=256,
                 lstm_num_layers=2,      # LSTM 레이어 수를 2로 증가 (dropout 적용 가능)
                 transformer_d_model=256,
                 transformer_nhead=4,
                 transformer_num_layers=1,
                 num_classes=2):
        super().__init__()

        # 1) CNN: ResNet50 (마지막 FC 제거)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_output_dim = 2048
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2) AutoEncoder: 2048 -> ae_latent_dim -> 2048
        self.ae_encoder = nn.Sequential(
            nn.Linear(self.cnn_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, ae_latent_dim),
            nn.ReLU()
        )
        self.ae_decoder = nn.Sequential(
            nn.Linear(ae_latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.cnn_output_dim)
        )
        
        # 3) LSTM: num_layers=2, dropout=0.2 (dropout는 layer 사이에 적용됨)
        self.lstm = nn.LSTM(
            input_size=ae_latent_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=0.2,  # 레이어 간 dropout
            batch_first=True
        )
        
        # 4) Transformer Encoder: dropout=0.1 추가
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model * 4,
            dropout=0.1,      # dropout 적용
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers
        )
        
        # 5) 최종 분류 FC
        self.fc_cls = nn.Linear(transformer_d_model, num_classes)

    def forward(self, x):
        """
        입력: x의 shape = [B, seq, C, H, W]
        출력:
          logits: [B, num_classes]
          feat_orig: 원본 CNN 특징 (재구성 대상, [B*seq, 2048])
          feat_recon: AutoEncoder 재구성 특징 ([B*seq, 2048])
        """
        B, seq, C, H, W = x.shape
        
        # CNN 특징 추출
        x = x.view(B * seq, C, H, W)
        feat_orig = self.cnn(x)                # [B*seq, 2048, 1, 1]
        feat_orig = feat_orig.view(B * seq, -1)  # [B*seq, 2048]
        
        # AutoEncoder: Encoder -> Decoder
        latent = self.ae_encoder(feat_orig)      # [B*seq, ae_latent_dim]
        feat_recon = self.ae_decoder(latent)     # [B*seq, 2048]
        
        # 시퀀스 구성: [B, seq, ae_latent_dim]
        latent_seq = latent.view(B, seq, -1)
        
        # LSTM 처리: 출력 shape [B, seq, lstm_hidden_dim]
        lstm_out, _ = self.lstm(latent_seq)
        
        # Transformer Encoder 적용: 입력 [B, seq, lstm_hidden_dim] → 출력 동일 shape
        trans_out = self.transformer(lstm_out)
        
        # 최종 분류: 마지막 타임스텝의 출력 사용
        final_out = trans_out[:, -1, :]         # [B, lstm_hidden_dim]
        logits = self.fc_cls(final_out)         # [B, num_classes]
        
        return logits, feat_orig, feat_recon

# 간단한 테스트
if __name__ == "__main__":
    model = CNNAE_LSTM_Transformer()
    dummy_input = torch.randn(2, 30, 3, 224, 224)  # 배치 2, 시퀀스 30, 3채널, 224x224
    logits, feat_orig, feat_recon = model(dummy_input)
    print("Logits shape:", logits.shape)
    print("Original feature shape:", feat_orig.shape)
    print("Reconstructed feature shape:", feat_recon.shape)
