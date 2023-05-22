from django.conf.urls import url
from . import views

urlpatterns = [
    url("register", views.register),
    url("login", views.login),
    url("logout", views.logout),
    url("profile_edit", views.profile_edit),
    url("test", views.test),
    url('update_yanzheng', views.update_yanzheng),
]