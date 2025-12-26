"""
Flask Web Application for Environment Image Interpreter
Provides web interface for uploading and analyzing images
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
from pathlib import Path
from werkzeug.utils import secure_filename
import cv2
import base64

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.detection.object_detector import ObjectDetector
from src.classification.scene_classifier import SceneClassifier
from src.interpretation.environment_interpreter import EnvironmentInterpreter
from src.utils.helpers import FileHandler, Logger, PerformanceMonitor

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results/output_images'

# Ensure directories exist
FileHandler.ensure_directory(app.config['UPLOAD_FOLDER'])
FileHandler.ensure_directory(app.config['RESULT_FOLDER'])

# Initialize components
logger = Logger('logs/web_app.log')
preprocessor = ImagePreprocessor()
detector = None  # Lazy loading
classifier = None  # Lazy loading
interpreter = EnvironmentInterpreter()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_models():
    """Lazy initialization of ML models"""
    global detector, classifier
    
    if detector is None:
        logger.info("Initializing Object Detector...")
        detector = ObjectDetector(confidence_threshold=0.25)
    
    if classifier is None:
        logger.info("Initializing Scene Classifier...")
        classifier = SceneClassifier(use_gpu=True)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(Path(filename).stem.split('_')[-1]) if '_' in filename else 0)
        unique_filename = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {unique_filename}")
        
        # Initialize models
        initialize_models()
        
        # Process image
        monitor = PerformanceMonitor()
        
        # Preprocessing
        monitor.start()
        processed_img, original_img = preprocessor.preprocess_for_detection(filepath)
        preprocessed_tensor = preprocessor.preprocess_for_classification(filepath)
        preprocess_time = monitor.stop('Preprocessing')
        
        # Object Detection
        monitor.start()
        detection_results = detector.detect_objects(
            filepath,
            save_output=True,
            output_path=os.path.join(app.config['RESULT_FOLDER'], f'detected_{unique_filename}')
        )
        detection_time = monitor.stop('Object Detection')
        
        # Scene Classification
        monitor.start()
        classification_results = classifier.classify_scene(preprocessed_tensor)
        classification_time = monitor.stop('Scene Classification')
        
        # Environment Interpretation
        monitor.start()
        interpretation = interpreter.interpret_environment(
            detection_results,
            classification_results,
            detailed=True
        )
        interpretation_time = monitor.stop('Interpretation')
        
        # Prepare response
        response_data = {
            'success': True,
            'filename': unique_filename,
            'results': {
                'interpretation': interpretation['interpretation'],
                'scene_type': interpretation['scene_type'],
                'primary_scene': interpretation['primary_scene'],
                'object_count': interpretation['object_count'],
                'risk_level': interpretation['risk_level'],
                'activities': interpretation['suggested_activities'],
                'use_cases': interpretation['use_cases'],
                'confidence': interpretation['confidence_score']
            },
            'detections': detection_results['detections'],
            'performance': {
                'preprocessing': f"{preprocess_time:.3f}s",
                'detection': f"{detection_time:.3f}s",
                'classification': f"{classification_time:.3f}s",
                'interpretation': f"{interpretation_time:.3f}s",
                'total': f"{preprocess_time + detection_time + classification_time + interpretation_time:.3f}s"
            },
            'annotated_image': f'/results/detected_{unique_filename}'
        }
        
        logger.info(f"Processing completed for {unique_filename}")
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result images"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')


@app.route('/api/status')
def status():
    """API endpoint to check system status"""
    return jsonify({
        'status': 'online',
        'models_loaded': {
            'object_detector': detector is not None,
            'scene_classifier': classifier is not None
        }
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("INTELLIGENT ENVIRONMENT IMAGE INTERPRETER")
    print("Web Application Starting...")
    print("=" * 60)
    
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)