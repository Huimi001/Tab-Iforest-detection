"""wjhproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from user.views import login, register, index
from datamanage.views import updata
urlpatterns = [
    path('admin', admin.site.urls),  # 系统默认创建的
    path('', login),
    path('index', index),  # 用于打开注册页面
    path('updata', updata),  # 用于打开注册页面
    path('', include('user.urls')),
    path('', include('datamanage.urls')),
    path('', include('projectmanage.urls')),
    path('', include('reminder.urls'))
]
# path('login/', login),  # 用于打开登录页面
# path('register/', register),  # 用于打开注册页面
# path('register/save', save),  # 输入用户名密码后交给后台save函数处理
# path('login/query', query),  # 输入用户名密码后交给后台query函数处理