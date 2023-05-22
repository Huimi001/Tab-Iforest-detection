
from django.db import models

# Create your models here.
class project(models.Model):
    """project信息模型"""

    # 收货地址
    project_name = models.CharField(max_length=100, unique=True)
    loss_url = models.CharField(max_length=100)
    bili_url = models.CharField(max_length=100)
    output_url = models.CharField(max_length=100)
    project_acc = models.CharField(max_length=100)
    project_recall = models.CharField(max_length=100)
    project_prec = models.CharField(max_length=100)
    project_auc = models.CharField(max_length=100)
    project_time = models.CharField(max_length=100)
    user_id = models.CharField(max_length=100)
    # 收货人

    def __str__(self):
        return self.name