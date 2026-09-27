import os
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .apps import CoreConfig
from src.utils.helpers import Logger, PerformanceMonitor
from pathlib import Path
from django.utils.text import slugify
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from .forms import SignUpForm
from .models import AnalysisResult
import time
import uuid
import datetime

logger = Logger('logs/django_app.log')

@login_required
def index(request):
    """Main dashboard for environment analysis"""
    return render(request, 'core/index.html')

@login_required
def history(request):
    """View analysis history for the current user"""
    analyses = AnalysisResult.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'core/history.html', {'analyses': analyses})

@login_required
def clear_history(request):
    """Delete all analysis history for the current user"""
    if request.method == 'POST':
        AnalysisResult.objects.filter(user=request.user).delete()
        return redirect('history')
    return redirect('history')

@login_required
def download_analysis(request, analysis_id):
    """Download analysis data as JSON"""
    try:
        analysis = AnalysisResult.objects.get(id=analysis_id, user=request.user)
        data = {
            'id': analysis.id,
            'scene_type': analysis.scene_type,
            'interpretation': analysis.interpretation,
            'object_count': analysis.object_count,
            'risk_level': analysis.risk_level,
            'confidence_score': analysis.confidence_score,
            'date': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        }
        response = JsonResponse(data)
        response['Content-Disposition'] = f'attachment; filename="analysis_{analysis.id}.json"'
        return response
    except AnalysisResult.DoesNotExist:
        return JsonResponse({'error': 'Analysis not found'}, status=404)

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = SignUpForm()
    return render(request, 'core/signup.html', {'form': form})

@login_required
def delete_account(request):
    """Permanently delete user account and all associated data"""
    if request.method == 'POST':
        user = request.user
        user.delete()
        return redirect('login')
    return redirect('index')

@login_required
@csrf_exempt
def upload_image(request):
    """Handle image upload and processing"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
        
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
            
        file = request.FILES['file']
        
        if file.name == '':
            return JsonResponse({'error': 'No file selected'}, status=400)
            
        # Save uploaded file
        filename = slugify(Path(file.name).stem) + Path(file.name).suffix
        # Ensure name is unique
        timestamp = str(int(Path(filename).stem.split('_')[-1]) if '_' in filename else 0)
        unique_filename = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
        
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, unique_filename)
        
        with open(filepath, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
                
        logger.info(f"File uploaded: {unique_filename}")
        
        # Initialize models (lazy)
        detector = CoreConfig.get_detector()
        classifier = CoreConfig.get_classifier()
        interpreter = CoreConfig.get_interpreter()
        preprocessor = CoreConfig.get_preprocessor()
        
        # Process image
        monitor = PerformanceMonitor()
        
        # Preprocessing
        monitor.start()
        # Note: we pass the absolute filepath
        processed_img, original_img = preprocessor.preprocess_for_detection(filepath)
        preprocess_time = monitor.stop('Preprocessing')
        
        # Object Detection
        monitor.start()
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results', 'output_images')
        os.makedirs(result_dir, exist_ok=True)
        detection_output_path = os.path.join(result_dir, f'detected_{unique_filename}')
        
        detection_results = detector.detect_objects(
            filepath,
            save_output=True,
            output_path=detection_output_path
        )
        detection_time = monitor.stop('Object Detection')
        
        # Scene Classification
        monitor.start()
        classification_results = classifier.classify_scene(filepath)
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
            'annotated_image': f'/media/results/output_images/detected_{unique_filename}'
        }

        # Save to database
        AnalysisResult.objects.create(
            user=request.user,
            image=f'uploads/{unique_filename}',
            annotated_image_path=response_data['annotated_image'],
            interpretation=response_data['results']['interpretation'],
            scene_type=response_data['results']['primary_scene'],
            object_count=response_data['results']['object_count'],
            risk_level=response_data['results']['risk_level'],
            confidence_score=response_data['results']['confidence']
        )
        
        logger.info(f"Processing completed and saved for {unique_filename}")
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

def status(request):
    """API endpoint to check system status"""
    return JsonResponse({
        'status': 'online',
        'models_loaded': {
            'object_detector': CoreConfig.detector is not None,
            'scene_classifier': CoreConfig.classifier is not None
        }
    })
