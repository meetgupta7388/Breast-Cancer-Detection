from flask import Flask, redirect, render_template, request, make_response, session, abort, jsonify, url_for
import secrets
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
from datetime import timedelta
import os
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import joblib

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

import gdown

def download_if_not_exists(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url} using gdown...")
        gdown.download(url, filename, quiet=False)


# Load the VGG19 model
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras import layers
    
    # Define the missing Cast layer
    class Cast(layers.Layer):
        def __init__(self, dtype=None, **kwargs):
            super(Cast, self).__init__(**kwargs)
            self.dtype_value = dtype
        
        def call(self, inputs):
            return tf.cast(inputs, self.dtype_value or tf.float32)
        
        def get_config(self):
            config = super(Cast, self).get_config()
            config.update({'dtype': self.dtype_value})
            return config
    
    # Define Google Drive direct download links
    vgg_model_url = 'https://drive.google.com/uc?id=1feKw46IU9su0wmxSh1mBocFXNhq63J1T'
    xgb_model_url = 'https://drive.google.com/uc?id=1qaFaKXekNJhPpjnETQAkjx8zoSQ1zB9b'
    scaler_url    = 'https://drive.google.com/uc?id=1utTu6tCqtC-56tmf2LtvXvc2M-HKnpVl'


    # Download models if not present
    download_if_not_exists(vgg_model_url, 'breast_cancer_vgg19.h5')
    download_if_not_exists(xgb_model_url, 'xgb_model.pkl')
    download_if_not_exists(scaler_url, 'scaler.pkl')

    # Get the absolute path to the model file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'breast_cancer_vgg19.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        model = None
    else:
        print(f"Attempting to load model from {model_path}")
        # Register custom objects with your custom Cast layer
        custom_objects = {
            'Cast': Cast
        }
        
        # Load model with custom objects and explicit input shape
        model = load_model(model_path, 
                         custom_objects=custom_objects,
                         compile=False)
                         
        # Ensure model is ready for inference by running a test prediction
        test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model.predict(test_input)
        
        print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    import traceback
    traceback.print_exc()
    model = None

# Firebase configuration
app.config['FIREBASE_API_KEY'] = os.getenv('FIREBASE_API_KEY')
app.config['FIREBASE_AUTH_DOMAIN'] = os.getenv('FIREBASE_AUTH_DOMAIN')
app.config['FIREBASE_PROJECT_ID'] = os.getenv('FIREBASE_PROJECT_ID')
app.config['FIREBASE_STORAGE_BUCKET'] = os.getenv('FIREBASE_STORAGE_BUCKET')
app.config['FIREBASE_MESSAGING_SENDER_ID'] = os.getenv('FIREBASE_MESSAGING_SENDER_ID')
app.config['FIREBASE_APP_ID'] = os.getenv('FIREBASE_APP_ID')

# Configure session cookie settings for localhost
app.config['SESSION_COOKIE_SECURE'] = False  # Set to False for localhost
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Firebase Admin SDK setup
import json

firebase_creds_json = os.getenv("FIREBASE_CREDS_JSON")
if firebase_creds_json:
    firebase_creds = json.loads(firebase_creds_json)
    cred = credentials.Certificate(firebase_creds)
else:
    raise Exception("FIREBASE_CREDS_JSON is not set")

firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID')
})
db = firestore.client()

# Make config available to all templates
@app.context_processor
def inject_config():
    return dict(config=app.config)

########################################
""" Authentication and Authorization """

# Decorator for routes that require authentication
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated
        if 'user' not in session:
            return redirect(url_for('login'))
        
        else:
            return f(*args, **kwargs)
        
    return decorated_function


@app.route('/auth', methods=['POST'])
def authorize():
    token = request.headers.get('Authorization')
    if not token or not token.startswith('Bearer '):
        return jsonify({"error": "Unauthorized"}), 401

    token = token[7:]  # Strip off 'Bearer ' to get the actual token

    try:
        decoded_token = auth.verify_id_token(token)
        session['user'] = {
            'uid': decoded_token['uid'],
            'email': decoded_token.get('email', ''),
            'name': decoded_token.get('name', '')
        }
        session.permanent = True  # Make the session permanent
        return jsonify({"success": True, "redirect": "/dashboard"}), 200
    except Exception as e:
        print(f"Auth error: {str(e)}")
        return jsonify({"error": str(e)}), 401


#####################
""" Public Routes """

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html')

@app.route('/signup')
def signup():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('signup.html')


@app.route('/reset-password')
def reset_password():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('forgot_password.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove the user from session
    response = make_response(redirect(url_for('login')))
    response.set_cookie('session', '', expires=0)  # Optionally clear the session cookie
    return response

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

##############################################
""" Private Routes (Require authorization) """

@app.route('/dashboard')
@auth_required
def dashboard():

    return render_template('dashboard.html')

@app.route('/profile')
@auth_required
def profile():
    return render_template('profile.html')

@app.route('/update-profile', methods=['POST'])
@auth_required
def update_profile():
    try:
        data = request.get_json()
        user_id = session['user']['uid']
        
        # Update user profile in Firestore
        user_ref = db.collection('users').document(user_id)
        user_ref.update({
            'first_name': data.get('first_name'),
            'last_name': data.get('last_name'),
            'age': int(data.get('age')),
            'gender': data.get('gender'),
            'phone': data.get('phone'),
            'address': data.get('address'),
            'medical_history': data.get('medical_history'),
            'updated_at': firestore.SERVER_TIMESTAMP
        })

        # Update session data
        session['user'].update({
            'first_name': data.get('first_name'),
            'last_name': data.get('last_name'),
            'age': int(data.get('age')),
            'gender': data.get('gender'),
            'phone': data.get('phone'),
            'address': data.get('address'),
            'medical_history': data.get('medical_history')
        })
        
        return jsonify({"message": "Profile updated successfully"}), 200
    except Exception as e:
        print(f"Profile update error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/upload-profile-image', methods=['POST'])
@auth_required
def upload_profile_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # Get user ID from session
            user_id = session['user']['uid']
            
            # Upload to Firebase Storage
            bucket = storage.bucket()
            blob = bucket.blob(f'profile_images/{user_id}/{file.filename}')
            blob.upload_from_string(
                file.read(),
                content_type=file.content_type
            )
            
            # Get the public URL
            blob.make_public()
            image_url = blob.public_url
            
            # Update user profile in Firestore
            user_ref = db.collection('users').document(user_id)
            user_ref.update({
                'profile_image': image_url,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
            
            # Update session data
            session['user']['profile_image'] = image_url
            
            return jsonify({"image_url": image_url}), 200
        else:
            return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        print(f"Image upload error: {str(e)}")
        return jsonify({"error": str(e)}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model input size (224x224 for VGG19)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Ensure array has the right shape (224, 224, 3)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:,:,:3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_breast_cancer(image):
    if model is None:
        raise Exception("Model not loaded. Please check the model file and try again.")
        
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Print shape for debugging
        print(f"Processed image shape: {processed_image.shape}")
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Get probability and class
        probability = float(prediction[0][0])
        class_name = "Malignant" if probability > 0.5 else "Benign"
        confidence = probability if class_name == "Malignant" else 1 - probability
        
        return {
            "class": class_name,
            "probability": probability,
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        # Add more detailed error information
        import traceback
        traceback.print_exc()
        raise Exception(f"Prediction failed: {str(e)}")
    
@app.route('/predict', methods=['POST'])
@auth_required
def predict():
    try:
        analysis_type = request.form.get('analysis_type') or request.json.get('analysis_type')
        
        if analysis_type == 'xray':
            if model is None:
                print("Error: Model is not loaded")
                return jsonify({"error": "Model not loaded. Please try again later."}), 500

            if 'image' not in request.files:
                print("Error: No image file in request")
                return jsonify({"error": "No image file provided"}), 400

            file = request.files['image']
            if file.filename == '':
                print("Error: Empty filename")
                return jsonify({"error": "No selected file"}), 400

            if not allowed_file(file.filename):
                print(f"Error: Invalid file type - {file.filename}")
                return jsonify({"error": "File type not allowed"}), 400

            try:
                # Read and process the image
                image = Image.open(file)
                print("Image loaded successfully")
                
                # Make prediction
                prediction = predict_breast_cancer(image)
                print(f"Prediction made: {prediction}")
                
                # Add analysis type to prediction
                prediction['analysis_type'] = 'xray'
                
                # Get user ID from session
                user_id = session['user']['uid']
                
                # Save prediction to Firestore
                prediction_ref = db.collection('predictions').document()
                prediction_ref.set({
                    'user_id': user_id,
                    'prediction': prediction,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                print("Prediction saved to Firestore")
                
                return jsonify({
                    "prediction": prediction
                }), 200
            except Exception as e:
                print(f"Error processing image or making prediction: {str(e)}")
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
                
        elif analysis_type == 'parameter':
            try:
                # Get parameters from request
                parameters = request.json.get('parameters')
                if not parameters:
                    return jsonify({"error": "No parameters provided"}), 400
                
                # Load XGBoost model
                xgb_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_model.pkl')
                if not os.path.exists(xgb_model_path):
                    print(f"Error: Model file not found at {xgb_model_path}")
                    return jsonify({"error": "Parameter analysis model not found"}), 500
                
                try:
                    xgb_model = joblib.load(xgb_model_path)
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    return jsonify({"error": f"Error loading model: {str(e)}"}), 500
                
                # Prepare features in the correct order
                try:
                    features = [
                        float(parameters['worst_concave_points']),
                        float(parameters['worst_perimeter']),
                        float(parameters['mean_concave_points']),
                        float(parameters['worst_radius']),
                        float(parameters['mean_perimeter']),
                        float(parameters['worst_area']),
                        float(parameters['mean_radius']),
                        float(parameters['mean_area']),
                        float(parameters['mean_concavity'])
                    ]
                    print(f"Features prepared: {features}")
                except Exception as e:
                    print(f"Error preparing features: {str(e)}")
                    return jsonify({"error": f"Error preparing features: {str(e)}"}), 400
                
                # Make prediction
                try:
                    prediction_proba = xgb_model.predict_proba([features])[0]
                    probability = float(prediction_proba[1])
                    print(f"Prediction probability: {probability}")
                except Exception as e:
                    print(f"Error making prediction: {str(e)}")
                    return jsonify({"error": f"Error making prediction: {str(e)}"}), 500
                
                # Determine risk level
                if probability >= 0.7:
                    risk_class = "High"
                elif probability >= 0.4:
                    risk_class = "Medium"
                else:
                    risk_class = "Low"
                
                prediction = {
                    "class": risk_class,
                    "probability": probability,
                    "confidence": round(probability * 100, 2),
                    "analysis_type": "parameter"
                }
                
                # Get user ID from session
                user_id = session['user']['uid']
                
                # Save prediction to Firestore
                try:
                    prediction_ref = db.collection('predictions').document()
                    prediction_ref.set({
                        'user_id': user_id,
                        'prediction': prediction,
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    print("Prediction saved to Firestore successfully")
                except Exception as e:
                    print(f"Error saving to Firestore: {str(e)}")
                    # Continue even if saving fails
                
                return jsonify({
                    "prediction": prediction
                }), 200
                
            except Exception as e:
                print(f"Error in parameter analysis: {str(e)}")
                return jsonify({"error": f"Error in parameter analysis: {str(e)}"}), 500
                
        else:
            return jsonify({"error": "Invalid analysis type"}), 400
            
    except Exception as e:
        print(f"Unexpected error in predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-predictions')
@auth_required
def get_predictions():
    try:
        user_id = session['user']['uid']
        # Simplified query without ordering by timestamp
        predictions = db.collection('predictions').where('user_id', '==', user_id).get()
        
        prediction_list = []
        for pred in predictions:
            pred_data = pred.to_dict()
            pred_data['id'] = pred.id
            prediction_list.append(pred_data)
            
        # Sort predictions by timestamp in Python
        prediction_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
        return jsonify({"predictions": prediction_list}), 200
    except Exception as e:
        print(f"Error fetching predictions: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/xray-analysis')
@auth_required
def xray_analysis():
    return render_template('xray_analysis.html')

@app.route('/parameter-analysis')
@auth_required
def parameter_analysis():
    return render_template('parameter_analysis.html')

@app.route('/history')
@auth_required
def history():
    return render_template('history.html')

if __name__ == '__main__':
    app.run(debug=True)
