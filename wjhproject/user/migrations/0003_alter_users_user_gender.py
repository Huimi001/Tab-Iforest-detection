# Generated by Django 3.2.4 on 2021-06-08 14:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0002_auto_20210525_1148'),
    ]

    operations = [
        migrations.AlterField(
            model_name='users',
            name='user_gender',
            field=models.CharField(default='man', max_length=32),
        ),
    ]