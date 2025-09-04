from django.db import models
from django.contrib.auth.models import User

# Basic user profile for potential future extensions
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)