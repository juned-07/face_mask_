from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import sqlite3
import os
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Setup SQLite database connection with username as primary key
db_path = 'face_mask.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    face_without_mask BLOB NOT NULL,
    face_with_mask BLOB NOT NULL
)
''')
conn.commit()

# Initialize Haar cascade globally
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    """Detect faces using Haar cascades."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def get_largest_face(image):
    """Return the largest detected face in the image."""
    faces = detect_face(image)
    if len(faces) == 0:
        return None
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    return largest_face

def detect_mask_status(image):
    """
    Check if a mask is present by analyzing the lower half of the face.
    Returns a tuple (mask_detected, face_coordinates).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return False, None
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    (x, y, w, h) = largest_face
    face_roi = image[y:y + h, x:x + w]
    # Use the lower half of the face for mask detection
    lower_face = face_roi[h // 2:h, :]
    hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([25, 170, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = lower_face.shape[0] * lower_face.shape[1]
    skin_ratio = skin_pixels / total_pixels
    print(f"Mask detection: skin ratio = {skin_ratio:.3f}")
    # If less than 60% of the lower face is detected as skin, assume a mask is present.
    mask_detected = skin_ratio < 0.60
    return mask_detected, largest_face

def compare_faces(known_face_blob, unknown_face, threshold=0.55):
    """
    Compare the stored face image with the newly captured face region.
    Both images are converted to grayscale, resized, equalized,
    and then compared using histogram correlation and SSIM.
    Returns True if the weighted combined score exceeds the threshold.
    """
    # Decode the stored face image from blob
    known_face_array = np.frombuffer(known_face_blob, dtype=np.uint8)
    known_face_img = cv2.imdecode(known_face_array, cv2.IMREAD_COLOR)

    # Convert to grayscale
    known_face_gray = cv2.cvtColor(known_face_img, cv2.COLOR_BGR2GRAY)
    unknown_face_gray = cv2.cvtColor(unknown_face, cv2.COLOR_BGR2GRAY)

    # Resize images to a standard size for consistency
    standard_size = (100, 100)
    known_face_gray = cv2.resize(known_face_gray, standard_size)
    unknown_face_gray = cv2.resize(unknown_face_gray, standard_size)

    # Apply histogram equalization to reduce lighting differences
    known_face_gray = cv2.equalizeHist(known_face_gray)
    unknown_face_gray = cv2.equalizeHist(unknown_face_gray)

    # Calculate histograms and normalize
    hist_known = cv2.calcHist([known_face_gray], [0], None, [256], [0, 256])
    hist_unknown = cv2.calcHist([unknown_face_gray], [0], None, [256], [0, 256])
    cv2.normalize(hist_known, hist_known, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_unknown, hist_unknown, 0, 1, cv2.NORM_MINMAX)
    correlation = cv2.compareHist(hist_known, hist_unknown, cv2.HISTCMP_CORREL)

    # Calculate Structural Similarity Index (SSIM)
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(known_face_gray, unknown_face_gray)
    except ImportError:
        ssim_score = correlation  # Fallback

    # Compute weighted combined score: 0.4 for correlation, 0.6 for SSIM
    combined_score = 0.4 * correlation + 0.6 * ssim_score
    print(f"Combined face match score: {combined_score:.3f} (Histogram: {correlation:.3f}, SSIM: {ssim_score:.3f})")
    return combined_score > threshold

def base64_to_image(base64_string):
    """Convert a base64 string to an OpenCV image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@app.route('/')
def home():
    return redirect(url_for('login_choice'))


@app.route('/login_choice')
def login_choice():
    return render_template('login_choice.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Process unmasked registration image
        image_without_mask = base64_to_image(request.form['face_without_mask'])
        largest_face_unmasked = get_largest_face(image_without_mask)
        if largest_face_unmasked is None:
            return "<script>alert('No face detected in the unmasked image. Please try again.'); window.location.href='/register';</script>"
        (x, y, w, h) = largest_face_unmasked
        face_region_unmasked = image_without_mask[y:y + h, x:x + w]

        # Process masked registration image and verify mask presence
        image_with_mask = base64_to_image(request.form['face_with_mask'])
        mask_detected, largest_face_masked = detect_mask_status(image_with_mask)
        if not mask_detected or largest_face_masked is None:
            return "<script>alert('Mask not detected. Please wear a mask and try again.'); window.location.href='/register';</script>"
        (x_m, y_m, w_m, h_m) = largest_face_masked
        face_region_masked = image_with_mask[y_m:y_m + h_m, x_m:x_m + w_m]

        # Encode cropped face regions for storage
        ret1, buf1 = cv2.imencode('.jpg', face_region_unmasked)
        ret2, buf2 = cv2.imencode('.jpg', face_region_masked)
        if not ret1 or not ret2:
            return "<script>alert('Error processing images. Please try again.'); window.location.href='/register';</script>"

        cursor.execute("INSERT INTO users (username, password, face_without_mask, face_with_mask) VALUES (?, ?, ?, ?)",
                       (username, password, buf1.tobytes(), buf2.tobytes()))
        conn.commit()

        return "<script>alert('Registration successful! Please login.'); window.location.href='/login_choice';</script>"
    return render_template('register.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    # Retrieve login_mode (either 'masked' or 'unmasked')
    login_mode = request.form.get('login_mode', 'unmasked')

    image = base64_to_image(request.form['face_image'])
    largest_face_login = get_largest_face(image)
    if largest_face_login is None:
        return "<script>alert('No face detected. Please try again.'); window.location.href='/login_choice';</script>"
    (x, y, w, h) = largest_face_login
    face_region_login = image[y:y + h, x:x + w]

    cursor.execute("SELECT face_without_mask, face_with_mask FROM users WHERE username=? AND password=?",
                   (username, password))
    user = cursor.fetchone()
    if user:
        # Set a lower threshold for unmasked login to allow slight variations
        threshold = 0.45 if login_mode == 'unmasked' else 0.55
        reference_face_blob = user[0] if login_mode == 'unmasked' else user[1]
        if compare_faces(reference_face_blob, face_region_login, threshold):
            session['username'] = username
            return redirect(url_for('welcome'))
        else:
            return "<script>alert('Face does not match records. Please try again.'); window.location.href='/login_choice';</script>"
    return "<script>alert('Invalid username or password.'); window.location.href='/login_choice';</script>"


@app.route('/welcome')
def welcome():
    return f"<h1>Welcome, {session.get('username', 'User')}!</h1>"


if __name__ == '__main__':
    app.run(debug=True)
