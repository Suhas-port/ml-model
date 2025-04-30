from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
from datetime import datetime
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
import moviepy.video.fx.all as vfx

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
WATERMARKED_FOLDER = 'watermarked'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(WATERMARKED_FOLDER):
    os.makedirs(WATERMARKED_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['WATERMARKED_FOLDER'] = WATERMARKED_FOLDER

# Load the ML model
model_path = 'models/mesonet_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found! Check the models folder.")
model = load_model(model_path)

def create_watermark(video_width, video_height, duration):
    # Create background for watermark
    watermark_width = video_width // 3
    watermark_height = video_height // 6
    
    # Create semi-transparent black background
    background = ColorClip(size=(watermark_width, watermark_height), 
                         color=(0, 0, 0))
    background = background.set_opacity(0.7)
    background = background.set_duration(duration)
    
    # Create text watermark
    text = TextClip("DEEPFAKE DETECTED", 
                   fontsize=30, 
                   color='red',
                   font='Arial-Bold')
    text = text.set_duration(duration)
    
    # Position text in the center of background
    text = text.set_position(('center', 'center'))
    
    # Combine background and text
    watermark = CompositeVideoClip([background, text])
    
    return watermark

def process_video_with_watermark(input_path, output_path):
    try:
        # Load the video with audio
        video = VideoFileClip(input_path)
        
        # Create watermark
        watermark = create_watermark(video.w, video.h, video.duration)
        
        # Position watermark at bottom left
        watermark = watermark.set_position(('left', 'bottom'))
        
        # Combine video and watermark
        final_video = CompositeVideoClip([video, watermark])
        
        # Write the final video with audio
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # Close clips to free resources
        video.close()
        watermark.close()
        final_video.close()
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 30  # Analyze every 30th frame
    frame_count = 0
    fake_count = 0
    real_count = 0
    results = []

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

                results.append({
                    "frame": frame_count,
                    "label": label,
                    "score": float(fake_score)
                })

            frame_count += 1
    finally:
        cap.release()

    final_decision = "FAKE" if fake_count > real_count else "REAL"
    
    return {
        "frame_results": results,
        "total_fake_frames": fake_count,
        "total_real_frames": real_count,
        "final_decision": final_decision
    }

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Analyze the video
            results = analyze_video(temp_path)
            
            # If video is fake, create watermarked version
            if results['final_decision'] == 'FAKE':
                watermarked_filename = f"watermarked_{filename}"
                watermarked_path = os.path.join(app.config['WATERMARKED_FOLDER'], watermarked_filename)
                process_video_with_watermark(temp_path, watermarked_path)
                results['watermarked_path'] = watermarked_filename
            
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['WATERMARKED_FOLDER'], filename),
        as_attachment=True,
        mimetype='video/mp4'
    )

if __name__ == '__main__':
    app.run(debug=True) 