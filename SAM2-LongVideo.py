# Import necessary libraries
import os
import yaml
import torch
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor
import yolo_trainset
# Function to display a mask on the plot
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        # Generate a random color with transparency
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Use a color map for consistent coloring
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Function to display points on the plot
def show_points(coords, labels, ax, marker_size=200):
    # Separate positive and negative points based on labels
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    # Plot positive points in green and negative points in red
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Function to handle mouse clicks on the plot
def on_click(event):
    global labels1, points1,ax
    if event.inaxes:
        # Ask for the label (1 for positive, 0 for negative)
        label = int(input("Enter label for this point (1 for positive, 0 for negative): "))
        labels1.append(label)
        points1.append([event.xdata, event.ydata])
        # Show the clicked point on the image with appropriate color
        color = 'green' if label == 1 else 'red'
        ax.scatter(event.xdata, event.ydata, color=color, marker='*', s=200, edgecolor='white', linewidth=1.25)
        plt.draw()

def save_result(out_frame_idx):
    plt.figure(figsize=(6, 4))
    plt.title(f"{frame_names[out_frame_idx]}")
    frame_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
    plt.imshow(frame_image)
    if out_frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    # Save the current frame as an image
    plt.axis('off')  # Turn off axes
    plt.tight_layout(pad=0)
    output_path = os.path.join(output_dir, frame_names[out_frame_idx])
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

def propagate_through_subsequence():
    global subsequence, model_yolo, inference_state, video_segments, predictor, save_sam2_results, use_pretrained_YOLO
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        # out_frame_idx start from the prompt idx then decrease
        if subsequence > 1:
            segment_id = out_frame_idx + (subsequence - 2) * propagate_folder_size + train_number
        else:
            segment_id = out_frame_idx
        video_segments[segment_id] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if save_sam2_results ==True:
            save_result(segment_id)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # out_frame_idx start from the prompt idx then increase
        if subsequence > 1:
            segment_id = out_frame_idx + (subsequence - 2) * propagate_folder_size + train_number
        else:
            segment_id = out_frame_idx
        video_segments[segment_id] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        } 
        if save_sam2_results == True:
            save_result(segment_id)
        if subsequence == 1 and use_pretrained_YOLO == False:
            yolo_trainset.generate_yolo_training_dataset(out_frame_idx, frame_names, video_segments)

def init_propagate_state():
    global inference_state, use_pretrained_YOLO, labels1, points1, ax, wrong, wrong_subsequence, save_yolo_results
    if subsequence == 1 and use_pretrained_YOLO == False:
        # Initialize the inference state with the predictor model
        inference_state = predictor.init_state(video_path=os.path.join(target_dir, f'folder_{subsequence}'))
        prompt_frame_idx = 0
        if config["click"] == True:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.title(f"frame {prompt_frame_idx}")
            ax.imshow(Image.open(os.path.join(video_dir, frame_names[prompt_frame_idx])))

            # Connect the click event to the function
            cid = fig.canvas.mpl_connect('button_press_event', on_click)

            # Show the plot for interaction
            plt.show()
        else:
            points1.extend(config["coordinates"])
            labels1.extend(config["labels"])

        # Convert points and labels to numpy arrays for further processing
        points1 = np.array(points1, dtype=np.float32)
        labels1 = np.array(labels1, dtype=np.int32)

        # Process the collected points and add new points to the predictor
        if len(points1) != 0:
            _, _, _ = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=prompt_frame_idx,
                obj_id=1,
                points=points1,
                labels=labels1,
            )

    else:
        inference_state = predictor.init_state(video_path=os.path.join(target_dir, f'folder_{subsequence}'))
        prompt_frame_idx = 0
        for index, file_name in enumerate(os.listdir(os.path.join(target_dir, f'folder_{subsequence}'))):
            file_path = os.path.join(target_dir, f'folder_{subsequence}', file_name)
            results = model_yolo(file_path)
            yolo_conf1 = 0
            yolo_bbox1 = []
            if subsequence == 1:
                frame_id = index
            else:
                frame_id = (subsequence - 2)*propagate_folder_size + index + train_number
            for result in results:
                if save_yolo_results == True:
                    result.save(filename=os.path.join(output_yolo_dir, f"result{frame_names[frame_id]}"))  # save to disk
                conf = result.boxes.conf
                bbox = result.boxes.xyxy
                # print("out_frame_idx:",file_name)
                # print("conf:", conf)
                # print("bbox:", bbox)
                if bbox.numel() != 0:
                    top_conf, top_index = torch.max(conf, dim=0)
                    if top_conf > config["confidence_threshold"]:
                        bbox1 = bbox[top_index]
                        yolo_bbox1 = bbox1.cpu().numpy()
                        yolo_conf1 = top_conf
                        break
            if yolo_conf1 > config["confidence_threshold"]:
                prompt_frame_idx = index
                break
        # print("prompt_frame_idx", prompt_frame_idx)
        if len(yolo_bbox1) == 0:# 处理非空列表
            wrong = 1
            wrong_subsequence.append(subsequence)
        elif yolo_bbox1.size != 0:
            wrong = 0
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt_frame_idx,
                obj_id=1,
                box = yolo_bbox1,
            )

    yolo_trainset.init_dataset()

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    use_pretrained_YOLO = config["use_pretrained_YOLO"]
    # Define paths for model checkpoint and configuration
    yolo_model = config["yolo_model"]
    model_yolo = YOLO(yolo_model)
    sam2_checkpoint = config["sam2_checkpoint"]
    sam2_cfg = config["sam2_cfg"]

    # Set training parameters
    train_number = config["train_number"] # Use the first 'train_number' frames to train YOLO
    train_epoch = config["train_epoch"]
    train_patience = config["train_patience"]
    yolo_target_size = config["yolo_target_size"] # The size of YOLO training set images

    # Directory containing video frames
    video_dir = config["video_dir"]

    # Set whether to output the result as images
    save_sam2_results = config["save_sam2_results"] # Save the segmentation results of SAM2
    save_yolo_results = config["save_yolo_results"] # Save the segmentation results of YOLO

    # Directory to save output images
    output_dir = config["output_dir"]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_yolo_dir = config["output_yolo_dir"]
    if os.path.exists(output_yolo_dir):
        shutil.rmtree(output_yolo_dir)
    os.makedirs(output_yolo_dir, exist_ok=True)

    # Get all JPEG frame names in the directory and sort them by frame index
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    total_frames = len(frame_names)

    # Distribute frames into folders
    target_dir = "notebooks/videos/sorted_folders"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    propagate_folder_size = config["propagate_folder_size"]
    subsequence = config["subsequence"] # The folder to start
    # When `subsequence = 1`, start from training YOLO.(The 'model_yolo' should be YOLO("yolo11s.pt"))
    # When `subsequence > 1`, skip the training process and directly use pretrained weights to segment from the specified `subsequence`.(The 'model_yolo' should be your pretrained weights)
    if (total_frames - train_number) % propagate_folder_size == 0:
        folder_amount = (total_frames - train_number) // propagate_folder_size + 1
    else:
        folder_amount = (total_frames - train_number) // propagate_folder_size + 2
    print("The number of folders:",folder_amount)
    new_folder = os.path.join(target_dir, f'folder_1')
    os.makedirs(new_folder, exist_ok=True)
    for image_file in frame_names[0:train_number]:
        shutil.copy(os.path.join(video_dir, image_file), new_folder)
    for i in range(train_number, total_frames, propagate_folder_size):
        folder_number = (i - train_number) // propagate_folder_size + 2  # Calculate folder number
        new_folder = os.path.join(target_dir, f'folder_{folder_number}')
        os.makedirs(new_folder, exist_ok=True)
        for image_file in frame_names[i:i + propagate_folder_size]:
            shutil.copy(os.path.join(video_dir, image_file), new_folder)

    # Check if the CUDA device has a major version of 8 or higher and enable TF32 for matrix multiplication
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build the video predictor model
    predictor = build_sam2_video_predictor(sam2_cfg, sam2_checkpoint)
    video_segments = {}  # video_segments contains the per-frame segmentation results
    wrong_subsequence = []

    # Run propagation throughout the video
    wrong = 0
    yaml_file_path = os.path.join(video_dir, "data", "data.yaml")
    while True:
        if subsequence > 1 or use_pretrained_YOLO == True:
            # Enable automatic casting for mixed precision training with CUDA
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if subsequence == 1 and use_pretrained_YOLO == False:
            # Initialize lists to store points and labels
            points1 = []
            labels1 = []
        init_propagate_state()
        if wrong == 0:
            propagate_through_subsequence()
        if subsequence == 1 and use_pretrained_YOLO == False:
            # Train YOLO11
            yolo_trainset.split_yolo_training_dataset()
            model_yolo.train(data=yaml_file_path, epochs=train_epoch, imgsz=yolo_target_size, patience=train_patience)
            shutil.rmtree(os.path.join(video_dir, "data"))
        subsequence += 1
        if (subsequence - 1) == folder_amount:
            break
    shutil.rmtree(target_dir)
    if wrong_subsequence != []:
        print("Pay attention! The following subsequences do not have prompts and were not segmented, possibly due to insufficient YOLO training or the absence of target objects in the subsequences.", wrong_subsequence)