from flask import Flask, render_template, request, redirect, url_for, session, Response, flash
import mysql.connector
import cv2
import face_recognition
import numpy as np
import dlib

# Load the pre-trained models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL connection setup
db = mysql.connector.connect(
    host="localhost",
    user="admin",
    password="1234",
    database="face_authentication_db"
)

# Global variable to store known face encoding
known_face_encoding = None

def capture_face_encodings():
    video_capture = cv2.VideoCapture(0)
    face_encodings = []
    for i in range(5):
        ret, frame = video_capture.read()
        face_locations = face_detector(frame)
        if face_locations:
            shape = shape_predictor(frame, face_locations[0])
            face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))
            face_encodings.append(face_encoding)
    video_capture.release()
    return face_encodings

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global known_face_encoding
        
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open video stream.")
            return
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            face_locations = face_detector(frame)
            if face_locations:
                shape = shape_predictor(frame, face_locations[0])
                face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))
                
                if known_face_encoding is not None:
                    distance = face_recognition.face_distance([np.array(known_face_encoding)], face_encoding)[0]
                    confidence = (1 - distance) * 100
                    text = f"Confidence: {confidence:.2f}%"
                    print(text)  # Debugging print
                    
                    # Draw the confidence level on the frame
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        video_capture.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Capture face encoding using webcam
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        video_capture.release()
        
        if face_encodings:
            face_encoding = face_encodings[0].tolist()
            
            cursor = db.cursor()
            cursor.execute("INSERT INTO users (username, password, face_encoding) VALUES (%s, %s, %s)",
                           (username, password, str(face_encoding)))
            db.commit()
            return redirect(url_for('home'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        cursor = db.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        if result and result[0] == password:
            flash("Authenticated with username and password")
            return redirect(url_for('dashboard'))
        else:
            flash("Authentication failed. Please check your username and password.")
    
    return render_template('login.html')

@app.route('/login_face', methods=['POST'])
def login_face():
    global known_face_encoding  # Use the global variable to store known face encoding
    
    cursor = db.cursor()
    cursor.execute("SELECT face_encoding FROM users")
    results = cursor.fetchall()
    
    for result in results:
        known_face_encoding = np.array(eval(result[0]))  # Convert string back to NumPy array
        face_encodings = capture_face_encodings()
        distances = [face_recognition.face_distance([known_face_encoding], encoding)[0] for encoding in face_encodings]
        
        if distances:  # Check if the distances list is not empty
            avg_distance = sum(distances) / len(distances)
            confidence = (1 - avg_distance) * 100  # Convert distance to confidence percentage
            
            match_count = sum(distance <= 0.4 for distance in distances)  # Use a threshold to determine matches
            
            if match_count >= 3:  # Require at least 3 matches out of 5 frames
                flash(f"Authenticated with confidence level: {confidence:.2f}%")
                return redirect(url_for('dashboard'))  # Redirect to dashboard
    
    flash("Authentication failed with face recognition.")
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    cursor = db.cursor(buffered=True)  # Use buffered cursor
    cursor.execute("SELECT username FROM users")
    users = cursor.fetchall()
    cursor.close()
    return render_template('dashboard.html', users=users)


if __name__ == '__main__':
    app.run(debug=True)