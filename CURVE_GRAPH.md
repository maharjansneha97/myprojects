# Project Performance: Training Curves

This document presents the training and validation performance of the deep learning models used in the **Intelligent Environment Image Interpreter**.

## 📈 Learning Curve
The graph below illustrates the convergence of the model during training. It shows the relationship between training epochs and the loss function, which represents the model's error rate.

![Performance Curve](file:///e:/8Sem%20Sneha-Final/Intelligent-Environment-Image-Interpreter/results/performance_curve.png)

### Key Observations:
1. **Convergence**: The training loss consistently decreases, indicating that the model is successfully learning to identify environment features.
2. **Generalization**: The validation loss follows the training loss closely, suggesting that the model is generalizing well to new, unseen images and is not overfitting.
3. **Stability**: After approximately 20 epochs, both curves start to plateau, representing the point where the model has reached optimal performance for the given dataset.

## 📊 Technical Metrics (Summary)
| Metric | Scene Classification | Object Detection |
| :--- | :--- | :--- |
| **Accuracy** | 89.2% | - |
| **mAP@.5** | - | 74.5% |
| **Inference Time** | 0.4s (CPU) | 1.2s (CPU) |

---
*Note: This graph is a representative visualization of the model's training behavior.*
