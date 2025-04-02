# SW-MAIN/deeplearning/training/train.py

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from deeplearning.models.model import CNNAE_LSTM_Transformer
from deeplearning.dataset_loader.falling_dataset import FallingDataset

def train_one_epoch(model, dataloader, optimizer, device, alpha=0.1, epoch_idx=0, num_epochs=1):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    total_loss, total_ce_loss, total_mse_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx % 5 == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch_idx+1}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} Elapsed: {elapsed:.2f}s")
            
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, feat_orig, feat_recon = model(inputs)
        ce_loss = ce_loss_fn(logits, labels)
        mse_loss = mse_loss_fn(feat_recon, feat_orig)
        loss = ce_loss + alpha * mse_loss

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        total_mse_loss += mse_loss.item() * batch_size

        _, preds = torch.max(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_ce = total_ce_loss / total_samples
    avg_mse = total_mse_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, avg_ce, avg_mse, accuracy

@torch.no_grad()
def validate_one_epoch(model, dataloader, device, alpha=0.1, epoch_idx=0, num_epochs=1):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    total_loss, total_ce_loss, total_mse_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx % 5 == 0:
            elapsed = time.time() - start_time
            print(f"[Val Epoch {epoch_idx+1}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} Elapsed: {elapsed:.2f}s")
            
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits, feat_orig, feat_recon = model(inputs)
        ce_loss = ce_loss_fn(logits, labels)
        mse_loss = mse_loss_fn(feat_recon, feat_orig)
        loss = ce_loss + alpha * mse_loss

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        total_mse_loss += mse_loss.item() * batch_size

        _, preds = torch.max(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_ce = total_ce_loss / total_samples
    avg_mse = total_mse_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, avg_ce, avg_mse, accuracy

def main():
    # 하이퍼파라미터
    num_epochs = 10
    learning_rate = 1e-4
    alpha = 0.1
    batch_size = 4
    seq_length = 30

    # 데이터 증강 및 전처리 (transform 파이프라인)
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset & DataLoader
    train_dataset = FallingDataset(
        root_dir="./dataset",  # 실제 전처리된 데이터 경로에 맞게 수정 (예: "./data/processed")
        list_file="./deeplearning/data_preprocessing/train_list.txt",
        transform=transform_pipeline,
        seq_length=seq_length
    )
    val_dataset = FallingDataset(
        root_dir="./dataset",
        list_file="./deeplearning/data_preprocessing/val_list.txt",
        transform=transform_pipeline,
        seq_length=seq_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 및 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    model = CNNAE_LSTM_Transformer(
        ae_latent_dim=256,
        lstm_hidden_dim=256,
        lstm_num_layers=2,       # 변경: LSTM 레이어 수를 2로 설정 (dropout 활성화)
        transformer_d_model=256,
        transformer_nhead=4,
        transformer_num_layers=1,
        num_classes=2
    ).to(device)

    # 옵티마이저: Weight Decay 추가
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # 학습률 스케줄러: ReduceLROnPlateau 적용
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    print("----- Training Start -----")
    for epoch in range(num_epochs):
        train_loss, train_ce, train_mse, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, alpha=alpha, epoch_idx=epoch, num_epochs=num_epochs
        )
        val_loss, val_ce, val_mse, val_acc = validate_one_epoch(
            model, val_loader, device, alpha=alpha, epoch_idx=epoch, num_epochs=num_epochs
        )

        print(f"[Epoch {epoch+1}/{num_epochs}]")
        print(f"  Train: Loss={train_loss:.4f} (CE={train_ce:.4f}, MSE={train_mse:.4f}), Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} (CE={val_ce:.4f}, MSE={val_mse:.4f}), Acc={val_acc:.4f}")
        
        # 스케줄러 업데이트: 검증 손실 기준
        scheduler.step(val_loss)

    save_path = "cnn_ae_lstm_transformer.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    main()
