import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model
model_path = 'models/mesonet_model.h5'
if not os.path.exists(model_path):
    print("Model file not found! Check the models folder.")
    exit()

model = load_model(model_path)

# Load video
video_path = 'deepfake_mark_zuckerberg.mp4'
if not os.path.exists(video_path):
    print("Video file not found!")
    exit()

cap = cv2.VideoCapture(video_path)

frame_skip = 30  # Analyze every 30th frame
frame_count = 0

fake_count = 0
real_count = 0

# Open the log file outside the loop to ensure it remains open
output_log = 'detection_results.txt'
log_file = open(output_log, 'w')
log_file.write("Frame-by-frame detection results:\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            resized_frame = cv2.resize(frame, (256, 256))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)

            prediction = model.predict(input_frame, verbose=0)
            fake_score = prediction[0][0]

            if fake_score > 0.5:
                fake_count += 1
                label = "Fake"
            else:
                real_count += 1
                label = "Real"

            log_file.write(f"Frame {frame_count}: {label} ({fake_score:.2f})\n")

        frame_count += 1
finally:
    cap.release()
    log_file.write("\n--- Final Decision ---\n")
    log_file.write(f"Total Fake frames: {fake_count}\n")
    log_file.write(f"Total Real frames: {real_count}\n")

    if fake_count > real_count:
        log_file.write("Result: The video is likely a DEEPFAKE (FAKE).\n")
    else:
        log_file.write("Result: The video is likely REAL.\n")

    log_file.close()

print(f"Detection results saved to {output_log}")