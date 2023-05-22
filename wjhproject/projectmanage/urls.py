from django.conf.urls import url
from . import views

urlpatterns = [
    url("startproject", views.startproject),
    url("runproject", views.runproject),
    url(r'^projectlist', views.projectlist),
    url(r'^p_preview/*', views.p_preview),
    url(r'^p_turnpage/*', views.p_turnpage),
]