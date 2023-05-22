from django.conf.urls import url
from . import views

urlpatterns = [
    url("updata", views.updata),
    url(r'^preview/*', views.preview),
    url(r'^datasetlist', views.datasetlist),
    url(r'^turnpage/*', views.turnpage),
    url(r'^delete/*', views.delete),
]