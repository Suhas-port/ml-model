import os
import cv2

def extract_frames_from_videos(input_folder, output_folder):
    # Supported video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    for filename in os.listdir(input_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in video_extensions:
            video_path = os.path.join(input_folder, filename)
            print(f"Trying to open: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                continue

            count = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_frame_{count}.jpg")
                cv2.imwrite(frame_filename, frame)
                count += 1

            cap.release()
            print(f"✅ {count} frames saved from {filename}")
        else:
            print(f"⚠️ Skipping non-video file: {filename}")

def main():
    categories = ['real', 'fake']
    for category in categories:
        input_folder = os.path.join('dataset', category)
        output_folder = os.path.join('dataset', category)
        os.makedirs(output_folder, exist_ok=True)
        extract_frames_from_videos(input_folder, output_folder)

if __name__ == "__main__":
    main()
