# Project Testing Checklist & Guide

This document provides a structured list of tests to ensure the **Intelligent Environment Image Interpreter** is functioning correctly across all modules.

---

## 🏗️ 1. Technical Smoke Tests (Command Line)
Before running the web app, verify the individual components are loading correctly.

### A. Scene Classifier Smoke Test
Run the following to verify the model weights and categories are loading:
```bash
python src/classification/scene_classifier.py
```
*   **Expected Result**: Output shows "✓ Model loaded successfully!" and lists the scene categories.

### B. Object Detector Smoke Test
Verify the Faster R-CNN model and labels:
```bash
python src/detection/object_detector.py
```
*   **Expected Result**: Output shows "✓ Custom Faster R-CNN loaded successfully" and lists the 20 VOC categories.

---

## 🖥️ 2. Web Application Testing
Tests for the user interface and overall integration.

### A. Core Functionality
- [ ] **User Registration**: Create a new account and verify redirect to dashboard.
- [ ] **Image Upload**: Upload a `.jpg` or `.png` file.
- [ ] **Image Processing**: Verify the loading spinner or progress while interpreting.
- [ ] **Results Display**: Check if the interpretation text, detection boxes, and performance metrics are visible.
- [ ] **Annotated Image Download**: Verify that clicking the result image downloads the annotated version.

### B. History Management
- [ ] **History Table**: Navigate to the history page and verify past uploads are listed.
- [ ] **View Past Result**: Click on a previous entry to see details.
- [ ] **Clear History**: Use the "Clear History" button and verify the table is empty.

---

## 🔬 3. Algorithm & Logic Testing
Validating the "intelligence" of the system.

- [ ] **Scene Consistency**: Test an obvious "Kitchen" image. Ensure detected objects (sink, chair, bottle) match the "Indoor/Kitchen" scene.
- [ ] **Tiling Inference**: Upload an image with small, distant objects. Verify if the system detects them (which uses the quadrant-tiling algorithm).
- [ ] **Risk Assessment**:
    - [ ] Upload an image with a car/vehicle (Expected: MEDIUM RISK).
    - [ ] Upload a crowded street image (Expected: HIGH RISK due to person density).
    - [ ] Upload a peaceful park image (Expected: LOW RISK).
- [ ] **Preprocessing Check**: Compare the contrast of the annotated image with the original to ensure CLAHE and Sharpening were applied.

---

## 🛡️ 4. Edge Case Testing
Testing the system's robustness.

- [ ] **Non-Image Upload**: Try uploading a `.txt` or `.pdf` (Expected: Should show an error message).
- [ ] **Large File Upload**: Test with a very high-resolution photo (e.g., 10MB+).
- [ ] **No Objects Detection**: Upload a plain white or black image (Expected: "No objects detected" message, scene: "Unknown").
- [ ] **Logout Security**: Try accessing the `/history/` page after logging out (Expected: Redirect to Login).

---

## 📈 5. Performance Benchmarks
Check the "Performance" section in the results dashboard.
- **Preprocessing**: Should be < 0.1s.
- **Detection**: Should be < 2.0s (CPU).
- **Classification**: Should be < 0.5s.
- **Total Time**: Should be reasonable for the production environment.
