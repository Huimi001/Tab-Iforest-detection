from django.db import models


# Create your models here.
class remind(models.Model):
    """dataset信息模型"""
    reminder_name = models.CharField(max_length=100, unique=True)
    reminder_ctime = models.CharField(max_length=100)
    reminder_detail = models.CharField(max_length=200)

    def __str__(self):
        return self.name