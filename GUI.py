import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import subprocess
import yaml
import time
import threading
def save_config():
    config = {
        'use_pretrained_YOLO': use_pretrained_YOLO_var.get(),
        'yolo_model': yolo_model_var.get(),
        'sam2_checkpoint': sam2_checkpoint_var.get(),
        'sam2_cfg': sam2_cfg_var.get(),
        'train_number': int(train_number_var.get()),
        'train_epoch': int(train_epoch_var.get()),
        'train_patience': int(train_patience_var.get()),
        'yolo_target_size': int(yolo_target_size_var.get()),
        'save_sam2_results': True,
        'save_yolo_results': save_yolo_results_var.get(),
        'video_dir': video_dir_var.get(),
        'object_name': 'target',
        'output_dir': output_dir_var.get(),
        'output_yolo_dir': output_yolo_dir_var.get(),
        'propagate_folder_size': int(propagate_folder_size_var.get()),
        'subsequence': int(subsequence_var.get()),
        'confidence_threshold': float(confidence_threshold_var.get()),
        'click': False,
        'coordinates' : [[x, y] for x, y in points],
        'labels' : labels
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False) 

# Function for selecting a directory
def select_directory(entry_var):
    path = folder_selected = filedialog.askdirectory()
    if path:
        entry_var.set(folder_selected) 

def select_pt(entry_var):
    file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pt")])  # Set the file types that can be selected

    if file_path:
        entry_var.set(file_path)

def select_yaml(entry_var):
    file_path = filedialog.askopenfilename(filetypes=[("Yaml Files", "*.yaml")])
    if file_path:
        entry_var.set(file_path)

def on_confirm():
    def task_for_subprocess():
        try:
            subprocess.run(["python", "SAM2-LongVideo.py"], check=True)
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "An error occurred when calling SAM2-LongVideo.py")

    def window_of_output():
        threading.Thread(target=lambda: open_output_window("Drag the progress bar to see segmentation result"), daemon=True).start()

    def window_of_wait(message):
        def close_status_window():
            status_window.destroy()
        # Create and display pop-up window
        status_window = tk.Toplevel(root)
        status_window.title("Segmenting")
        status_window.geometry("300x100")

        status_label = tk.Label(status_window, text=message, font=("Arial", 12))
        status_label.pack(padx=20, pady=20)

        status_window.update()

        # Start a timer to automatically close the window after 20 seconds
        window_timer = threading.Timer(20, close_status_window)
        window_timer.start()
    if use_pretrained_YOLO_var.get():
        save_config() 
        window_of_wait("Starting, please wait...")
        # Create and start two separate threads
        subprocess_thread = threading.Thread(target=task_for_subprocess, daemon=True)
        subprocess_thread.start()
        window_timer = threading.Timer(10, window_of_output)
        window_timer.start()

    else:
        prompts_window = open_prompts_selection_window()
        prompts_window.wait_window()
        window_of_wait("Training YOLO, please wait...")
        # Create and start two separate threads
        subprocess_thread = threading.Thread(target=task_for_subprocess, daemon=True)
        subprocess_thread.start()
        window_timer = threading.Timer(10, window_of_output)
        window_timer.start()

# Open frame selection interface
def open_prompts_selection_window():
    prompts_window = tk.Toplevel(root)
    prompts_window.title("Left-click to select a positive prompt point, right-click to select a negative prompt point.")

    # Get the path of the first frame image in the video directory
    video_dir = video_dir_var.get()
    first_frame_path = os.path.join(video_dir, "00001.jpg")

    if os.path.exists(first_frame_path):
        image = Image.open(first_frame_path)
        photo = ImageTk.PhotoImage(image)

        canvas = tk.Canvas(prompts_window, width=image.width, height=image.height)
        canvas.pack()

        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo

        canvas.bind("<Button-1>", lambda event: on_image_click(event, "positive", canvas))
        canvas.bind("<Button-3>", lambda event: on_image_click(event, "negative", canvas))

        confirm_button = tk.Button(prompts_window, text="Confirm Prompts and Start Training YOLO", command=lambda: confirm_segmentation(prompts_window))
        confirm_button.pack()

    else:
        tk.Label(prompts_window, text="There is not the '00001.jpg frame.").pack()
    return prompts_window

def on_image_click(event, point_type, canvas):
    # Get the coordinates of the click
    x, y = event.x, event.y
    points.append([x,y])
    if point_type == 'positive':
        labels.append(1)
        color = 'green'
    else:
        labels.append(0)
        color = 'red'
    print(f"Clicked at ({x}, {y}) as {point_type} point")
    canvas.create_text(x, y, text="★", fill=color, font=("Arial", 20))

def confirm_segmentation(window):
    save_config()
    window.destroy()

def get_max_image_number():
    files = os.listdir(output_dir_var.get())
    jpg_files = [f for f in files if f.endswith('.jpg')]
    numbers = []
    for f in jpg_files:
        try:
            number = int(f.split('.')[0])
            numbers.append(number)
        except ValueError:
            pass
    
    if numbers:
        return max(numbers)
    else:
        return None

def check_missing_frames(missing_label, config):
    all_files = [
        p for p in os.listdir(output_dir_var.get())
        if os.path.splitext(p)[-1] in [".jpg"]
    ]
    all_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    missing_ranges = []
    if config['subsequence'] == 1:
        expected_frame = 1
    else:
        expected_frame = (config['subsequence'] - 2) * config['propagate_folder_size'] + config['train_number'] + 1
    
    for i, file in enumerate(all_files):
        frame_num = int(os.path.splitext(file)[0])
        if frame_num != expected_frame:
            missing_ranges.append([expected_frame,frame_num - 1])
            expected_frame = frame_num + 1
        else:
            expected_frame += 1
    if missing_ranges != []:
        # missing_label.config(text=f"Missing Frames(didn't detect the target):\n {missing_ranges}")
        grouped_ranges = [
            ", ".join([f"{start}-{end}" for start, end in missing_ranges[i:i + 6]])
            for i in range(0, len(missing_ranges), 8)
        ]
        formatted_ranges = "\n".join(grouped_ranges)
        missing_label.config(text=f"Missing Frames(didn't detect the target):\n{formatted_ranges}")
    else:
        missing_label.config(text="")


def update_progress_and_subsequence(sequence_label, missing_label, config):
    frame_names = [
        p for p in os.listdir(config['video_dir'])
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    total_frames = len(frame_names)
    total_subsequences = (total_frames - config['train_number']) // config['propagate_folder_size'] + 2
    current_subsequence = config['subsequence'] 
    # Update every 1 second
    count = 0
    while True:
        # current_subsequence value
        max_number = get_max_image_number()
        if max_number!=None:
            if max_number <= config['train_number']:
                current_subsequence = 1
            else:
                current_subsequence = (max_number - config['train_number']) // config['propagate_folder_size'] + 2
        # Update subsequence display
        if config['use_pretrained_YOLO'] == False and current_subsequence == 1:
            sequence_label.config(text="Training YOLO, please wait.")
        else:
            sequence_label.config(text=f"Processing Subsequence {current_subsequence}/{total_subsequences}")

        time.sleep(1)
        count += 1
        if count == 5:
            count = 0
            check_missing_frames(missing_label, config)

# Open a new window to display the processing subsequence, progress bar, and images
def open_output_window(message):
    output_window = tk.Toplevel(root)
    output_window.title("segmentation Result")
    output_window.geometry("800x600")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create label to display the message
    label = tk.Label(output_window, text=message, font=("Arial", 14))
    label.pack(pady=(25,5))

    frame_names = [
        p for p in os.listdir(video_dir_var.get())
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    total_frames = len(frame_names)
    total_subsequences = (total_frames - config['train_number']) // config['propagate_folder_size'] + 1
    
    if config['use_pretrained_YOLO'] == False and config['subsequence'] == 1:
        sequence_label = tk.Label(output_window, text="Training YOLO, please wait.", font=("Arial", 12))
    else:
        sequence_label = tk.Label(output_window, text=f"Processing Subsequence {config['subsequence']}/{total_subsequences}", font=("Arial", 12))
    sequence_label.pack(pady=(10,0))

    if config['subsequence'] == 1:
        progress = tk.DoubleVar(value=1)
    else:
        v = config['propagate_folder_size'] * (config['subsequence'] - 2) + config['train_number'] + 1
        progress = tk.DoubleVar(value=v)
    progress_bar = tk.Scale(output_window, from_=1, to=total_frames, orient="horizontal", variable=progress, sliderlength=20, length=600)
    progress_bar.pack(pady=0)
    
    # Display missing frame information
    missing_label = tk.Label(output_window, text=f"", font=("Arial", 10))
    missing_label.pack(pady=(10,0))

    # Label to display images
    image_label = tk.Label(output_window)
    image_label.pack(pady=20)

    # Image update function
    def update_image(value, image_label):
        image_name = f"{int(value):05d}.jpg"
        image_path = os.path.join(output_dir_var.get(), image_name)
        image_label.image = None
        if os.path.exists(image_path):
            image = Image.open(image_path)
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo, text="")
            image_label.image = photo
        else:
            image_label.config(image=None, text="This frame hasn't been segmented", fg="red", font=("Arial", 12))
            image_label.image = None

    # Initially display the image for the current subsequence
    if config['subsequence'] == 1:
        segment_id = 1
    else:
        segment_id = (config['subsequence'] - 2) * config['propagate_folder_size'] + config['train_number'] + 1
    update_image(segment_id, image_label)

    # Update the image when the progress bar is moved
    def on_progress_change(value):
        update_image(int(value), image_label)

    progress_bar.bind("<Motion>", lambda event: on_progress_change(progress.get()))  # Update the image when sliding the progress bar
    # Start a thread to update progress and subsequence information
    threading.Thread(target=lambda: update_progress_and_subsequence(sequence_label, missing_label, config), daemon=True).start()

    return output_window


def on_use_pretrained_yolo_change():
    # If use_pretrained_yolo is set to False, set subsequence to 1 and disable the input box
    if not use_pretrained_YOLO_var.get():
        subsequence_var.set("1")  # Automatically set to 1
        subsequence_entry.config(state="disabled")  # Lock the subsequence input box
    else:
        subsequence_entry.config(state="normal")  # Restore the subsequence input box to be editable
        
# Create the main window
root = tk.Tk()
root.title("Parameter settings")

# Configuration options
use_pretrained_YOLO_var = tk.BooleanVar(value=True)
yolo_model_var = tk.StringVar(value="bestdog.pt")
sam2_checkpoint_var = tk.StringVar(value="checkpoints/sam2_hiera_large.pt")
sam2_cfg_var = tk.StringVar(value="sam2_hiera_l.yaml")
train_number_var = tk.StringVar(value="250")
train_epoch_var = tk.StringVar(value="300")
train_patience_var = tk.StringVar(value="50")
yolo_target_size_var = tk.StringVar(value="640")
save_yolo_results_var = tk.BooleanVar(value=False)
video_dir_var = tk.StringVar(value="notebooks/videos/dog")
output_dir_var = tk.StringVar(value="output_segment")
output_yolo_dir_var = tk.StringVar(value="yolo_output_bbox")
propagate_folder_size_var = tk.StringVar(value="50")
subsequence_var = tk.StringVar(value="1")
confidence_threshold_var = tk.StringVar(value="0.9")

use_pretrained_YOLO_var.trace("w", lambda *args: on_use_pretrained_yolo_change())

# Configure input fields and labels
fields = [
    ("Use Pretrained YOLO", use_pretrained_YOLO_var),
    ("YOLO Model", yolo_model_var),
    ("SAM2 Checkpoint", sam2_checkpoint_var),
    ("SAM2 Config", sam2_cfg_var),
    ("Image Number to Train YOLO", train_number_var),
    ("Train Epoch", train_epoch_var),
    ("Train Patience", train_patience_var),
    ("YOLO Training Size", yolo_target_size_var),
    ("Save YOLO Prompts", save_yolo_results_var),
    ("Video Directory", video_dir_var),
    ("Output Segmentation Directory", output_dir_var),
    ("Output YOLO Prompts Directory", output_yolo_dir_var),
    ("Subsequence Size", propagate_folder_size_var),
    ("the Subsequence to Start", subsequence_var),
    ("Confidence Threshold", confidence_threshold_var)
]
points=[]
labels=[]

for i, (label_text, var) in enumerate(fields):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
    
    if isinstance(var, tk.BooleanVar):
        entry = tk.Checkbutton(root, variable=var)
    else:
        entry = tk.Entry(root, textvariable=var)
    
    entry.grid(row=i, column=1, padx=10, pady=5)
    if label_text == "the Subsequence to Start":
        subsequence_entry = entry 

# Buttons for selecting video directory and output directory
yolo_model_button = tk.Button(root, text="select file", command=lambda: select_pt(yolo_model_var))
yolo_model_button.grid(row=1, column=2, padx=10, pady=5)

sam2_checkpoint_button = tk.Button(root, text="select file", command=lambda: select_pt(sam2_checkpoint_var))
sam2_checkpoint_button.grid(row=2, column=2, padx=10, pady=5)

sam2_cfg_button = tk.Button(root, text="select file", command=lambda: select_yaml(sam2_cfg_var))
sam2_cfg_button.grid(row=3, column=2, padx=10, pady=5)

video_dir_button = tk.Button(root, text="select directory", command=lambda: select_directory(video_dir_var))
video_dir_button.grid(row=9, column=2, padx=10, pady=5)

output_dir_button = tk.Button(root, text="select directory", command=lambda: select_directory(output_dir_var))
output_dir_button.grid(row=10, column=2, padx=10, pady=5)

output_yolo_dir_button = tk.Button(root, text="select directory", command=lambda: select_directory(output_yolo_dir_var))
output_yolo_dir_button.grid(row=11, column=2, padx=10, pady=5)

# Display the submit button
confirm_button = tk.Button(root, text="OK", command=on_confirm)
confirm_button.grid(row=len(fields), column=0, columnspan=3, pady=20)

root.mainloop()
