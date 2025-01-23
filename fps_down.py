import cv2
import os

# Open the video file
video_path = 'dog.avi'
video_name = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

# Get the original frame rate of the video
original_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"original_fps: {original_fps}")
n = 15
new_fps = original_fps / n

os.makedirs(video_name, exist_ok=True)
frame_count = 0
saved_frame_count = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Save one image every n frames (effect of reducing frame rate)
    if frame_count % n == 0:
        output_path = os.path.join(video_name, f'{saved_frame_count:05d}.jpg')
        cv2.imwrite(output_path, frame)
        saved_frame_count += 1
    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"new_fps: {new_fps}")
print(f"Save {saved_frame_count-1} frames in total")
