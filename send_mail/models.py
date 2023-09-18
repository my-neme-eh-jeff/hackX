from django.db import models
from datetime import datetime 
# Create your models here.
class EmotionTrack(models.Model):
    uuid = models.CharField(max_length=200)
    emotion = models.CharField(max_length=50)
    time = models.TimeField()
