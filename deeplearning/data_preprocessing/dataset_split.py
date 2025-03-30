# deeplearning/data_preprocessing/dataset_split.py

import os
import yaml
import random

def load_config(config_path="config/config.yaml"):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def split_videos(processed_dir, label_name, label_value,
                 train_ratio, val_ratio, test_ratio):
    """
    processed_dir/label_name/ 폴더 아래 있는 video_x 목록을
    train/val/test로 분할 후, (상대경로, label)을 튜플 형태 리스트로 반환.
    label_name = "falling" or "not_falling"
    label_value = 1 or 0
    """
    folder_path = os.path.join(processed_dir, label_name)
    video_folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    random.shuffle(video_folders)

    total_count = len(video_folders)
    train_cut = int(total_count * train_ratio)
    val_cut = int(total_count * (train_ratio + val_ratio))

    train_list = video_folders[:train_cut]
    val_list = video_folders[train_cut:val_cut]
    test_list = video_folders[val_cut:]

    # (상대경로, 라벨)
    # 예: ("falling/video_1", 1)
    train_data = [(f"{label_name}/{v}", label_value) for v in train_list]
    val_data = [(f"{label_name}/{v}", label_value) for v in val_list]
    test_data = [(f"{label_name}/{v}", label_value) for v in test_list]

    return train_data, val_data, test_data

def write_list_to_file(data_list, filename):
    """
    data_list: [("falling/video_1", 1), ("not_falling/video_2", 0), ...]
    filename: "train_list.txt" 등
    """
    with open(filename, 'w') as f:
        for path, label in data_list:
            f.write(f"{path} {label}\n")

def main():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    train_ratio = config["data"]["train_split_ratio"]
    val_ratio = config["data"]["val_split_ratio"]
    test_ratio = config["data"]["test_split_ratio"]

    # falling -> label=1
    falling_train, falling_val, falling_test = split_videos(
        processed_dir, "falling", 1,
        train_ratio, val_ratio, test_ratio
    )

    # not_falling -> label=0
    not_falling_train, not_falling_val, not_falling_test = split_videos(
        processed_dir, "not_falling", 0,
        train_ratio, val_ratio, test_ratio
    )

    # 두 라벨의 리스트를 합침
    train_data = falling_train + not_falling_train
    val_data = falling_val + not_falling_val
    test_data = falling_test + not_falling_test

    # 섞어주면 더 랜덤하게 섞일 수 있음
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # txt 파일로 기록
    write_list_to_file(train_data, "train_list.txt")
    write_list_to_file(val_data, "val_list.txt")
    write_list_to_file(test_data, "test_list.txt")

    print("Dataset split done!")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    main()
