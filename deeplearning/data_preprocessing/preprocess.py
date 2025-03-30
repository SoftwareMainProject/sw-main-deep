# deeplearning/data_preprocessing/preprocess.py

import os
import yaml
from PIL import Image

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_images(src_dir, dst_dir, target_size=(224,224)):
    """
    src_dir 안에 video_x 폴더들이 있고, 각 폴더 내에 frame들이 있음.
    dst_dir로 전처리(리사이즈 등) 후 복사/저장한다.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    video_folders = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    for folder in video_folders:
        src_folder_path = os.path.join(src_dir, folder)
        dst_folder_path = os.path.join(dst_dir, folder)
        os.makedirs(dst_folder_path, exist_ok=True)

        frame_files = os.listdir(src_folder_path)
        for frame_file in frame_files:
            src_file_path = os.path.join(src_folder_path, frame_file)
            dst_file_path = os.path.join(dst_folder_path, frame_file)

            try:
                with Image.open(src_file_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size)
                    img.save(dst_file_path)
            except Exception as e:
                print(f"Warning: Could not process {src_file_path}. Error: {e}")

def main():
    config = load_config()
    raw_falling = config["data"]["raw_falling_dir"]
    raw_not_falling = config["data"]["raw_not_falling_dir"]
    processed_dir = config["data"]["processed_dir"]

    w = config["data"]["img_width"]
    h = config["data"]["img_height"]

    # falling 전처리
    dst_falling = os.path.join(processed_dir, "falling")
    preprocess_images(raw_falling, dst_falling, (w,h))

    # not_falling 전처리
    dst_not_falling = os.path.join(processed_dir, "not_falling")
    preprocess_images(raw_not_falling, dst_not_falling, (w,h))

    print("Preprocessing Done!")

if __name__ == "__main__":
    main()
