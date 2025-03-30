# falling_dataset.py

import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image

class FallingDataset(Dataset):
    """
    PyTorch Dataset for 낙상 감지 영상 데이터셋.
    
    Parameters:
        root_dir (str): 전처리된 이미지가 저장된 상위 폴더 (예: "./data/processed")
                        이 폴더 하위에는 "falling"과 "not_falling" 등의 하위 폴더가 존재.
        list_file (str): 각 영상 폴더의 상대 경로와 라벨 정보를 담은 텍스트 파일 경로.
                         파일 내용 예시 (한 줄에 하나):
                             falling/video_1 1
                             not_falling/video_2 0
        labels_csv_path (str, optional): CSV 파일로 라벨 정보를 관리할 경우의 경로.
                         CSV 파일 예시:
                             video_name,label
                             falling/video_1,1
                             not_falling/video_2,0
                         만약 list_file에 라벨 정보가 이미 포함되어 있다면 사용하지 않아도 됩니다.
        transform (callable, optional): 이미지에 적용할 전처리 및 증강 파이프라인 (예: torchvision.transforms.Compose([...]))
        seq_length (int): 각 시퀀스(영상)에서 사용할 프레임 수. 부족하면 마지막 프레임을 반복해서 채움.
    """
    def __init__(self, root_dir, list_file, transform=None, seq_length=30, labels_csv_path=None):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length

        # 우선 list_file에서 영상 폴더와 라벨 정보를 읽어옵니다.
        # 형식: "falling/video_1 1" 과 같이 한 줄에 경로와 라벨이 공백으로 구분되어 기록되어 있어야 함.
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    video_folder, label_str = parts
                    label = int(label_str)
                    self.samples.append((video_folder, label))

        # 만약 CSV 파일을 별도로 사용하고 싶다면, list_file에는 영상 폴더만 기록되어 있고,
        # labels_csv_path를 통해 라벨 정보를 읽어와 매핑 딕셔너리를 만듭니다.
        if labels_csv_path is not None:
            self.samples = []  # CSV 정보로 덮어씁니다.
            # CSV 파일 형식: video_name,label
            with open(labels_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    video_folder = row["video_name"].strip()
                    label = int(row["label"])
                    self.samples.append((video_folder, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 샘플 정보: (video_folder, label)
        video_folder, label = self.samples[idx]
        folder_path = os.path.join(self.root_dir, video_folder)

        # 폴더 내 이미지 파일들을 시간 순서대로 정렬
        frame_files = sorted(os.listdir(folder_path))

        frames = []
        for img_name in frame_files:
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue

            # transform이 정의되어 있다면 적용 (예: Resize, ToTensor, Normalize, Data Augmentation 등)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # 시퀀스 길이가 부족하면 마지막 프레임을 반복해서 채우고,
        # 길면 앞부분만 사용해서 고정 길이 시퀀스로 만듭니다.
        if len(frames) < self.seq_length:
            while len(frames) < self.seq_length:
                frames.append(frames[-1])
        else:
            frames = frames[:self.seq_length]

        # frames를 [seq_length, C, H, W] 형태의 텐서로 변환
        frames_tensor = torch.stack(frames, dim=0)

        return frames_tensor, label

# 사용 예시
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # transform 파이프라인 정의 (예: 이미지 리사이즈, ToTensor, Normalize)
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset 생성 시, list_file에 영상 폴더와 라벨 정보가 기록되어 있는 파일 경로를 지정합니다.
    dataset = FallingDataset(
        root_dir="./data/processed",         # 전처리된 이미지 폴더의 상위 경로
        list_file="./deeplearning/data_preprocessing/train_list.txt",
        transform=my_transform,
        seq_length=30,
        labels_csv_path=None                 # list_file에 라벨 정보가 포함되어 있다면 None, 별도 CSV 사용 시 경로 지정
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_data, batch_labels in dataloader:
        print("Batch 데이터 크기:", batch_data.shape)  # 예: (2, 30, 3, 224, 224)
        print("Batch 라벨:", batch_labels)
        break
