import cv2
import os

def video_to_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 1  # Start naming frames from 1
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Construct the output file name
        frame_filename = os.path.join(output_folder, f"{frame_count:05d}.jpg")
        # Save the frame as an image file
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video object
    video_capture.release()
    print(f"The video has been split into {frame_count - 1} frames, saved in the '{output_folder}' folder.")


# Example usage
video_path = 'dog.avi'
output_folder = os.path.splitext(os.path.basename(video_path))[0]
video_to_frames(video_path, output_folder)
