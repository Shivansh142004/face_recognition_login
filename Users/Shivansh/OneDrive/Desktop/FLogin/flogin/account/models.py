from django.db import models
from django.contrib.auth.models import User

class UserImages(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    login_id = models.CharField(max_length=20, unique=True, blank=True, null=True)
    face_image = models.ImageField(upload_to="user_faces/")

    def __str__(self):
        return f"{self.user.username} ({self.login_id})"
