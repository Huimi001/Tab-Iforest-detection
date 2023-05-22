from django.conf.urls import url
from . import views

urlpatterns = [
    url("add_reminder", views.add_reminder),
    url(r'^rdelete/*', views.rdetele),
]