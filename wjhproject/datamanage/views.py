import os

from django.core.paginator import PageNotAnInteger, EmptyPage, Paginator
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.shortcuts import HttpResponse
from . import models
from django.contrib import messages
import pymysql
from pandas import read_csv
from .models import dataset
from django.utils import timezone
import pytz


def updata(request):
    if request.method == 'GET':
        return render(request, 'updata.html')
    if request.method == 'POST':
        File = request.FILES.get("csv_file", None)
        if File is None:
            return HttpResponse("没有需要上传的文件")
        else:
            # 打开特定的文件进行二进制的写操作
            # print(os.path.exists('/temp_file/'))
            with open("./datamanage/upload_files/%s" % File.name, 'wb+') as f:
                # 分块写入文件
                for chunk in File.chunks():
                    f.write(chunk)
            tz = pytz.timezone('Asia/Shanghai')
            # 返回时间格式的字符串
            now_time = timezone.now().astimezone(tz=tz)
            now_time_str = now_time.strftime("%Y.%m.%d %H:%M:%S")
            datasetname = File.name
            dataseturl = "B:/Projects/wjh/wjhproject/datamanage/upload_files/%s" % File.name
            models.dataset.objects.create(dataset_name=datasetname, dataset_url=dataseturl, dataset_ctime=now_time_str)
            messages.success(request, "数据集上传成功！")
            return render(request, 'updata.html')
    else:
        return render(request, 'updata.html')


def preview(request):
    id = request.GET.get('id')
    data = dataset.objects.get(dataset_name=id)
    data_set = read_csv(data.dataset_url, nrows=50)
    data = data_set.values[:, :]
    paginator = Paginator(data, 10)
    # 获取当前的页码数，默认为1
    page = request.GET.get("page", 1)
    # 把当前的页码数转换为整数类型
    currentPage = int(page)
    try:
        test_data = paginator.page(currentPage)  # 获取当前页码的记录
    except PageNotAnInteger:
        test_data = paginator.page(1)  # 如果用户输入的页码不是整数时,显示第1页的内容
    except EmptyPage:
        test_data = paginator.page(paginator.num_pages)  # 如果用户输入的页码不是整数时,显示第1页的内容
    request.session['datasetid'] = id
    return render(request, "previewtest3.html", locals())


def turnpage(request):
    id = request.session.get('datasetid')
    data = dataset.objects.get(dataset_name=id)
    data_set = read_csv(data.dataset_url, nrows=50)
    data = data_set.values[:, :]
    paginator = Paginator(data, 10)
    # 获取当前的页码数，默认为1
    page = request.GET.get("page", 1)
    # 把当前的页码数转换为整数类型
    currentPage = int(page)
    try:
        test_data = paginator.page(currentPage)  # 获取当前页码的记录
    except PageNotAnInteger:
        test_data = paginator.page(1)  # 如果用户输入的页码不是整数时,显示第1页的内容
    except EmptyPage:
        test_data = paginator.page(paginator.num_pages)  # 如果用户输入的页码不是整数时,显示第1页的内容
    request.session['datasetid'] = id
    return render(request, "previewtest3.html", locals())


def datasetlist(request):
    dataset_list = dataset.objects.all()
    context = {"dataset_list": dataset_list}
    return render(request, "datasetlist.html", context=context)


# 0


def delete(request):
    id = request.GET.get('id')
    data = dataset.objects.get(dataset_name=id)
    models.dataset.objects.get(dataset_name=id).delete()
    path_file = data.dataset_url
    del_files(path_file)
    dataset_list = dataset.objects.all()
    context = {"dataset_list": dataset_list}
    messages.success(request, "数据集删除成功！")
    return render(request, "datasetlist.html", context=context)


def del_files(path_file):
    os.remove(path_file)
