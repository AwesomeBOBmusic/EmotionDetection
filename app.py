from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import os
import cv2
import joblib
from skimage.feature import hog
import numpy as np
import zipfile

# Initialize the Flask application
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure the static/uploads directory exists
os.makedirs('static/uploads', exist_ok=True)

# Load the trained SVM model
model = joblib.load("emotion_classifier.pkl")

# Define the emotion labels in the same order used during training
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Extract the CK+48 dataset if not already extracted
def extract_dataset():
    dataset_zip = 'CK+48.zip'
    if os.path.exists(dataset_zip):
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Dataset extracted successfully!")

# Preprocess images
def preprocess_images(input_dir, output_dir, target_size=(128, 128), equalize_hist=False):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.png'):
                img_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(img_path)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized_img = cv2.resize(gray_img, target_size)
                    normalized_img = resized_img / 255.0
                    if equalize_hist:
                        normalized_img = cv2.equalizeHist(np.uint8(normalized_img * 255)) / 255.0
                    rel_path = os.path.relpath(img_path, input_dir)
                    save_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, np.uint8(normalized_img * 255))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

# Ensure dataset is extracted and preprocessed
extract_dataset()
preprocess_images('CK+48', 'Preprocessed', equalize_hist=True)
print("Image preprocessing complete!")

# Update the base directory for CK+48 dataset
base_dir = '/Users/aoy3/Desktop/-UniSem6/Vison/WebEmotions/CK+48'

# Iterate through all subfolders and process images
def process_all_images(base_dir):
    for emotion_folder in os.listdir(base_dir):
        emotion_path = os.path.join(base_dir, emotion_folder)
        if os.path.isdir(emotion_path):
            for image_file in os.listdir(emotion_path):
                image_path = os.path.join(emotion_path, image_file)
                if image_file.endswith('.png'):
                    print(f"Processing: {image_path}")
                    # Add your image processing logic here

# Call the function to process all images
process_all_images(base_dir)

def preprocess_and_predict(filepath):
    # Load the image in grayscale
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 128x128 (same size used in training)
    resized_img = cv2.resize(img, (128, 128))

    # Normalize pixel values to the [0, 1] range
    normalized_img = resized_img / 255.0

    # Extract HOG features from the preprocessed image
    features = hog(normalized_img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')

    # Predict the emotion label using the trained classifier
    prediction = model.predict([features])[0]
    predicted_emotion = emotion_labels[prediction]

    return predicted_emotion

# Real-time emotion detection
@app.route('/video_feed')
def video_feed():
    def generate():
        # Load the Haar Cascade face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open the video capture (webcam or video file)
        cap = cv2.VideoCapture(0)  # Change to a video file path if needed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Extract the face region
                face_img = gray[y:y+h, x:x+w]

                # Resize and normalize the face
                resized_face = cv2.resize(face_img, (128, 128))
                normalized_face = resized_face / 255.0

                # Extract HOG features
                features = hog(normalized_face, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')

                # Predict the emotion
                prediction = model.predict([features])[0]
                predicted_emotion = emotion_labels[prediction]

                # Draw a rectangle around the face and overlay the emotion label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to process and display video frames with real-time emotion detection
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return redirect(request.url)
    video = request.files['video']
    if video.filename == '':
        return redirect(request.url)
    if video:
        filename = secure_filename(video.filename)
        video_path = os.path.join('static/uploads', filename)
        video.save(video_path)

        def generate():
            # Load the Haar Cascade face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    # Extract the face region
                    face_img = gray[y:y+h, x:x+w]

                    # Resize and normalize the face
                    resized_face = cv2.resize(face_img, (128, 128))
                    normalized_face = resized_face / 255.0

                    # Extract HOG features
                    features = hog(normalized_face, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), block_norm='L2-Hys')

                    # Predict the emotion
                    prediction = model.predict([features])[0]
                    predicted_emotion = emotion_labels[prediction]

                    # Draw a rectangle around the face and overlay the emotion label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Encode the frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame as part of the response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            cap.release()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', video_feed_url=url_for('video_feed'))

# Use the exact Real-time Emotion Prediction logic from the notebook
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)  # Save in static/uploads
        file.save(filepath)

        # Step 6.2: Load the Haar Cascade face detector (face detection method)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Step 6.3: Load the trained SVM model from Step 4 using joblib
        model = joblib.load("emotion_classifier.pkl")

        # Step 6.4: Define the emotion labels in the same order used during training
        emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

        # Load the image
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 6.9: Detect faces in the grayscale frame using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Step 6.11: Extract the face region from the image (cropping)
            face_img = gray[y:y+h, x:x+w]

            # Step 6.12: Resize the face to 128x128 (same size used in training)
            resized_face = cv2.resize(face_img, (128, 128))

            # Step 6.13: Normalize pixel values to the [0, 1] range
            normalized_face = resized_face / 255.0

            # Step 6.14: Extract HOG features from the preprocessed face
            features = hog(normalized_face, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')

            # Step 6.15: Predict the emotion label using the trained classifier
            prediction = model.predict([features])[0]
            predicted_emotion = emotion_labels[prediction]

            # Step 6.16: Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Step 6.17: Overlay the predicted emotion label on the frame
            cv2.putText(img, predicted_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Save the processed image
        processed_filepath = os.path.join('static/uploads', f'processed_{filename}')
        cv2.imwrite(processed_filepath, img)

        return render_template('result.html', emotion="Emotion(s) detected", image_path=processed_filepath)

if __name__ == '__main__':
    app.run(debug=True)