import os
import numpy as np
import yaml
from PIL import Image
import random
import shutil
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Directory containing training data
video_dir = config["video_dir"]
yolo_target_size = config["yolo_target_size"] # The size of YOLO training set images

yolo_target_image = os.path.join(video_dir, "data", "images")
yolo_target_label = os.path.join(video_dir, "data", "labels")
train_images_dir = os.path.join(video_dir, "data", "train", "images")
train_labels_dir = os.path.join(video_dir, "data", "train", "labels")
valid_images_dir = os.path.join(video_dir, "data", "valid", "images")
valid_labels_dir = os.path.join(video_dir, "data", "valid", "labels")
test_images_dir = os.path.join(video_dir, "data", "test", "images")
test_labels_dir = os.path.join(video_dir, "data", "test", "labels")
yaml_file_path = os.path.join(video_dir, "data", "data.yaml")
name = config["object_name"]
data_yaml = {
    'path': os.path.join(os.getcwd(), video_dir, "data"),
    'train': os.path.join("train", "images"),
    'val': os.path.join("valid", "images"),
    'test': os.path.join("test", "images"),
    'nc': 1,
    'names': [name]
}

# Function to init YOLO training dataset
def init_dataset():
    if os.path.exists(os.path.join(video_dir, "data")):
        shutil.rmtree(os.path.join(video_dir, "data"))
    os.makedirs(yolo_target_image, exist_ok=True)
    os.makedirs(yolo_target_label, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    with open(yaml_file_path, 'w') as file:
        yaml.dump(data_yaml, file, default_flow_style=False)

def move_files(file_list, images_dest, labels_dest):
    for file in file_list:
        image_file = f"{file}.jpg"  # Adjust according to image file extension
        label_file = f"{file}.txt"
        shutil.move(os.path.join(yolo_target_image, image_file), os.path.join(images_dest, image_file))
        if os.path.exists(os.path.join(yolo_target_label, label_file)):  
            shutil.move(os.path.join(yolo_target_label, label_file), os.path.join(labels_dest, label_file))

def resize_image(image, original_width, original_height):
    image.thumbnail((yolo_target_size, yolo_target_size), Image.LANCZOS)  # Maintains aspect ratio scaling, largest side conforms to target size
    new_image = Image.new("RGB", (yolo_target_size, yolo_target_size),
                            (0, 0, 0))  # Create a completely black background image
    # Calculate the paste position to center the scaled image and paste the scaled image onto the black background.
    if original_width > original_height:
        y_offset = (yolo_target_size - image.height / image.width * yolo_target_size) // 2
        new_image.paste(image, (0, int(y_offset)))
    elif original_height > original_width:
        x_offset = (yolo_target_size - image.width / image.height * yolo_target_size) // 2
        new_image.paste(image, (int(x_offset), 0))
    else:
        new_image.paste(image, (0, 0))
    return new_image

def sam2_bbox_xywh(mask, original_width, original_height):
    y_indices, x_indices = np.where(mask > 0)
    if x_indices.any():
        xmin, xmax = x_indices.min(), x_indices.max()
        ymin, ymax = y_indices.min(), y_indices.max()
        center_x_old = (xmin + xmax) / 2.0
        center_y_old = (ymin + ymax) / 2.0
        width_old = xmax - xmin + 1
        height_old = ymax - ymin + 1
        if original_width > original_height:
            center_x = center_x_old / original_width
            center_y = (center_y_old / original_width * yolo_target_size + 0.5 * (
                    yolo_target_size - original_height / original_width * yolo_target_size)) / yolo_target_size
            width = width_old / original_width
            height = height_old / original_width
        elif original_height > original_width:
            center_x = (center_x_old / original_height * yolo_target_size + 0.5 * (
                    yolo_target_size - original_width / original_height * yolo_target_size)) / yolo_target_size
            center_y = center_y_old / original_height
            width = width_old / original_height
            height = height_old / original_height
        else:
            center_x = center_x_old / original_width
            center_y = center_y_old / original_height
            width = width_old / original_width
            height = height_old / original_height
        return [center_x, center_y, width, height]
    else:
        return []

def generate_yolo_training_dataset(generate_frame_idx, frame_names, video_segments):
    # Generate the image file
    frame_image = Image.open(os.path.join(video_dir, frame_names[generate_frame_idx]))
    width0, height0 = frame_image.size
    resized_image = resize_image(frame_image, width0, height0)
    yolo_target_path = os.path.join(video_dir, "data", "images", frame_names[generate_frame_idx])
    resized_image.save(yolo_target_path)
    # Generate the label file path
    file_number = frame_names[generate_frame_idx].split('.')[0]
    label_file_path = os.path.join(video_dir, "data", "labels", f"{file_number}.txt")
    for _, out_mask in video_segments[generate_frame_idx].items():
        out_mask = out_mask.squeeze()
        bbox_sam = sam2_bbox_xywh(out_mask, width0, height0)
        if bbox_sam != []:
            with open(label_file_path, 'a') as label_file:
                label_file.write("0 ")
                label_file.write(f"{bbox_sam[0]} {bbox_sam[1]} {bbox_sam[2]} {bbox_sam[3]}\n")


def split_yolo_training_dataset():
    file_names = [f.split('.')[0] for f in os.listdir(yolo_target_image)]
    random.shuffle(file_names)
    total_files = len(file_names)
    train_size = int(total_files * 0.7)
    valid_size = int(total_files * 0.2)
    train_files = file_names[:train_size]
    valid_files = file_names[train_size:train_size + valid_size]
    test_files = file_names[train_size + valid_size:]
    move_files(train_files, train_images_dir, train_labels_dir)
    move_files(valid_files, valid_images_dir, valid_labels_dir)
    move_files(test_files, test_images_dir, test_labels_dir)
