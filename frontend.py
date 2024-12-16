from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the saved model (using joblib for scikit-learn models)
model = tf.keras.models.load_model('EfficientNetv2.h5')

# Define the image preprocessing function


def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model's input size
    image = np.array(image)  # Convert image to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension (making it 2D)
    return image

# Convert image to base64 for displaying in HTML


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def adjust_keypoint_size(keypoints, scale_factor):
    for kp in keypoints:
        kp.size *= scale_factor  # Scale the keypoint size
    return keypoints

# Function to overlay keypoints on the image (SIFT + ORB)


def extract_orb_features(image):
    orb = cv2.ORB_create(
        nfeatures=25,           # Max number of keypoints
        # # scaleFactor=0.5,          # Image pyramid scaling factor
        # nlevels=8,                # Pyramid levels
        # edgeThreshold=10,         # Edge threshold size
        # firstLevel=0,             # Start at level 0 of pyramid
        # WTA_K=2,                  # Use two points for descriptor computation
        # scoreType=cv2.ORB_HARRIS_SCORE,  # Scoring method for keypoints
        # patchSize=31              # Descriptor patch size
    )
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = []
    return keypoints, descriptors


def extract_sift_features(image):
    sift = cv2.SIFT_create(
        nfeatures=0,            # Max number of keypoints. 0 means no limit
        # nOctaveLayers=10,
        # contrastThreshold=0.015,  # Threshold for filtering keypoints based on contrast
        # edgeThreshold=5,       # Threshold for filtering keypoints based on edge response
        # sigma=1.6
    )
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = []
    return keypoints, descriptors

def overlay_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Sobel filter
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # Extract features
    orb_keypoints, _ = extract_orb_features(img)
    orb_keypoints = adjust_keypoint_size(orb_keypoints, scale_factor=0.2)

    sift_keypoints, _ = extract_sift_features(img)

    # Draw keypoints
    img_with_keypoints = cv2.drawKeypoints(
        sobel_combined, orb_keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_with_keypoints = cv2.drawKeypoints(
        img_with_keypoints, sift_keypoints, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return img_with_keypoints


# Define class labels
class_labels = ['Cataract','a', 'glaucoma', 'normal']

# HTML template for the form and result
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retinal Disease Classification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f7f7f7;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 80%;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            margin-top: 50px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .form-container {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        input[type="file"] {
            padding: 10px;
            font-size: 16px;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        .result {
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            font-size: 18px;
            background-color: #eaf5e1;
            border: 1px solid #4CAF50;
            color: #4CAF50;
        }
        .image-container {
            text-align: center;
            margin-top: 30px;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Upload an Image for Retinal Disease Classification</h1>
    <div class="form-container">
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br><br>
            <input type="submit" value="Classify Image">
        </form>
    </div>
    
    {% if label %}
        <div class="result">
            <h2>Prediction: {{ label }}</h2>
            <p>Class Index: {{ class_index }}</p>
            <p>Confidence: {{ confidence }}%</p>
        </div>
        <div class="image-container">
            <h3>Image with SIFT + ORB Keypoints</h3>
            <img src="data:image/jpeg;base64,{{ image_with_keypoints_base64 }}" alt="Processed Image">
        </div>
    {% endif %}
</div>

</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(html_template)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Open image file and preprocess it
        image = Image.open(io.BytesIO(file.read()))

        image_with_keypoints = np.array(image)
        image_with_keypoints = overlay_keypoints(image_with_keypoints)
        image_for_prediction = preprocess_image(image)

        # Predict the class
        prediction = model.predict(image_for_prediction, batch_size=1)

        if prediction.ndim == 2:  # If the model returns probabilities
            predicted_class_index = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_index] * 100
        else:  # If the model returns a direct class label
            predicted_class_index = prediction[0]
            confidence = None

        label = f'{class_labels[predicted_class_index]}'

        # Convert image with keypoints to base64 for rendering
        image_with_keypoints_base64 = image_to_base64(
            Image.fromarray(image_with_keypoints))

        # Render the result with label, class index, confidence, and the image with keypoints overlay
        return render_template_string(html_template,
                                      label=label,
                                      class_index=predicted_class_index,
                                      confidence=confidence,
                                      image_with_keypoints_base64=image_with_keypoints_base64)


if __name__ == "__main__":
    app.run(debug=False)
