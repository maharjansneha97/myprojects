from django.db import models
from django.contrib.auth.models import User

class AnalysisResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    image = models.ImageField(upload_to='uploads/')
    annotated_image_path = models.CharField(max_length=255)
    interpretation = models.TextField()
    scene_type = models.CharField(max_length=100)
    object_count = models.IntegerField()
    risk_level = models.CharField(max_length=20)
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.scene_type} ({self.created_at})"
