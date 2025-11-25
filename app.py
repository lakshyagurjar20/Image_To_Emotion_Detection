import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load model and scaler
try:
    model = joblib.load('emotion_classifier_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Emotion mapping
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Happy',
    2: 'Neutral',
    3: 'Sad'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features(results):
    """Extract pose features from MediaPipe results"""
    features = []
    if results.pose_landmarks:
        # Extract specific landmarks (16 landmarks × 3 coordinates = 48 features)
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
                features.extend([lm.x, lm.y, lm.z])
    
    if len(features) != 48:
        return None
    return np.array(features).reshape(1, -1)

def process_image(image_path):
    """Process a single image and predict emotion"""
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error reading image"
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose.process(rgb_image)
    
    # Draw landmarks on image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    
    # Extract features and predict
    features = extract_features(results)
    if features is None:
        emotion = "No pose detected"
        confidence = 0
    else:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        emotion = EMOTION_LABELS.get(prediction, "Unknown")
        confidence = float(probabilities[prediction]) * 100
        
        # Add text to image
        cv2.putText(image, f'Emotion: {emotion} ({confidence:.1f}%)', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2)
    
    pose.close()
    
    # Save processed image
    output_filename = 'processed_' + os.path.basename(image_path)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    cv2.imwrite(output_path, image)
    
    return output_filename, emotion, confidence

def process_video(video_path):
    """Process video and predict emotions frame by frame"""
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video path
    output_filename = 'processed_' + os.path.basename(video_path)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    emotion_counts = {label: 0 for label in EMOTION_LABELS.values()}
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Extract features and predict
        features = extract_features(results)
        if features is not None:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            emotion = EMOTION_LABELS.get(prediction, "Unknown")
            confidence = float(probabilities[prediction]) * 100
            
            emotion_counts[emotion] += 1
            
            # Add text overlay
            cv2.putText(frame, f'Emotion: {emotion} ({confidence:.1f}%)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No pose detected', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    pose.close()
    
    # Calculate dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    return output_filename, dominant_emotion, emotion_counts, frame_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: mp4, avi, mov, mkv, jpg, jpeg, png'}), 400
    
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please ensure emotion_classifier_rf.pkl and scaler.pkl exist.'}), 500
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Determine if image or video
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    try:
        if file_ext in ['jpg', 'jpeg', 'png']:
            # Process image
            output_filename, emotion, confidence = process_image(filepath)
            return jsonify({
                'success': True,
                'type': 'image',
                'output_file': output_filename,
                'emotion': emotion,
                'confidence': confidence
            })
        else:
            # Process video
            output_filename, dominant_emotion, emotion_counts, frame_count = process_video(filepath)
            return jsonify({
                'success': True,
                'type': 'video',
                'output_file': output_filename,
                'dominant_emotion': dominant_emotion,
                'emotion_distribution': emotion_counts,
                'total_frames': frame_count
            })
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
