from django.db import models


# Create your models here.
class Users(models.Model):
    """用户信息模型"""

    # 订单编号
    user_id = models.CharField(max_length=100, unique=True)
    # 收货地址
    user_psd = models.CharField(max_length=100)
    # 收货人
    user_gender = models.CharField(max_length=32, default='man')
    # 联系电话
    user_tel = models.CharField(max_length=11)
    # 用户状态
    user_status = models.IntegerField(default=1)

    def __str__(self):
        return self.name

