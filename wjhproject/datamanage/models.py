from django.db import models


# Create your models here.
class dataset(models.Model):
    """dataset信息模型"""



    # 收货地址
    dataset_name = models.CharField(max_length=100, unique=True)
    dataset_url = models.CharField(max_length=100)
    dataset_ctime = models.CharField(max_length=100)
    # 收货人

    def __str__(self):
        return self.name