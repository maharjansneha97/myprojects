from django.apps import AppConfig
import sys
from pathlib import Path

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    
    detector = None
    classifier = None
    interpreter = None
    preprocessor = None

    def ready(self):
        # We don't load models here immediately to avoid slowing down management commands
        # instead we provide a method to get them
        pass

    @classmethod
    def get_detector(cls):
        if cls.detector is None:
            from src.detection.object_detector import ObjectDetector
            print("✓ Initializing Custom Faster R-CNN Object Detector...")
            cls.detector = ObjectDetector(confidence_threshold=0.3)
        return cls.detector

    @classmethod
    def get_classifier(cls):
        if cls.classifier is None:
            from src.classification.scene_classifier import SceneClassifier
            print("✓ Initializing Custom ResNet-18 Scene Classifier...")
            cls.classifier = SceneClassifier(use_gpu=True)
        return cls.classifier

    @classmethod
    def get_interpreter(cls):
        if cls.interpreter is None:
            from src.interpretation.environment_interpreter import EnvironmentInterpreter
            cls.interpreter = EnvironmentInterpreter()
        return cls.interpreter

    @classmethod
    def get_preprocessor(cls):
        if cls.preprocessor is None:
            from src.preprocessing.image_preprocessor import ImagePreprocessor
            cls.preprocessor = ImagePreprocessor()
        return cls.preprocessor
