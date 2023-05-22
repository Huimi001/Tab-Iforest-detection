from django.shortcuts import render, redirect
from . import models
from django.contrib import messages
# Create your views here.
from .models import remind


def add_reminder(request):
    if request.method == 'POST':
        remindername = request.POST.get('remindername')
        reminderctime = request.POST.get('reminderctime')
        reminderdetail = request.POST.get('reminderdetail')
        models.remind.objects.create(reminder_name=remindername, reminder_ctime=reminderctime, reminder_detail=reminderdetail)
        messages.success(request, "添加成功！")
        return redirect('/index')

def rdetele(request):
    id = request.GET.get('id')
    models.remind.objects.get(reminder_name=id).delete()
    messages.success(request, "便签删除成功！")
    return redirect('/index')




