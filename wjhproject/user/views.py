from django.http import HttpResponseRedirect
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.shortcuts import HttpResponse

from reminder.models import remind
from . import models
import pymysql
from django.contrib import messages
# 登录页面
# request.session.get('_auth_user_id')[获取用户id]
from PIL import Image, ImageDraw, ImageFont
import random
from io import BytesIO, StringIO


# 获取随机的样式颜色
from .models import Users


def get_random():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def get_code(request):
    img_obj = Image.new('RGB', (350, 30), get_random())  # 生成图片对象
    img_draw = ImageDraw.Draw(img_obj)  # 生成了一个画笔对象
    img_font = ImageFont.truetype('static/fonts/sylfaen.ttf', 30)  # 字体样式
    # 随机码的获取：
    code = ''
    for i in range(5):
        upper_str = chr(random.randint(65, 90))  # ascii码 大写字母
        lower_str = chr(random.randint(97, 122))  # ascii码 小写字母
        random_int = str(random.randint(0, 9))
        tmp = random.choice([upper_str, lower_str, random_int])  # 随机取值
        img_draw.text((i * 60 + 60, 0), tmp, get_random(), img_font)  # 文字展示到图片上
        code += tmp  # 一次结果
    print(code)
    request.session['code'] = code  # 存储
    io_obj = BytesIO()  # 内存内存储，读取
    img_obj.save('static/images/yanzheng.png', 'png')  # 保存，并选定格式
    return code


code = None


def login(request):
    global code
    if request.method == 'GET':
        code = get_code(request)
        return render(request, 'login.html')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        yanzheng = request.POST.get('yanzheng')
        if username == '':
            messages.success(request, "用户名不能为空！")
            return render(request, 'login.html')
        elif password == '':
            messages.success(request, "密码不能为空！")
            return render(request, 'login.html')
        else:
            if models.Users.objects.get(user_id=username) == '':
                messages.success(request, "用户名有误！")
                return render(request, 'login.html')
            else:
                if models.Users.objects.get(user_psd=password) == '':
                    messages.success(request, "密码有误！")
                    return render(request, 'login.html')
                else:
                    if yanzheng.lower() == code.lower():
                        request.session['username'] = username
                        request.session['is_login'] = True
                        messages.success(request, "登录成功！")
                        reminder_list = remind.objects.all()
                        context = {"reminder_list": reminder_list}
                                # return redirect('/index')
                        return render(request, 'index.html', context=context)
                    else:
                        messages.success(request, "验证码有误！")
                        return render(request, 'login.html')


def update_yanzheng(request):
    global code
    if request.method == 'GET':
        code = get_code(request)
        return render(request, 'login.html')


def index(request):
    # 指定要访问的页面，render的功能：讲请求的页面结果提交给客户端
    username = request.session.get('username', None)
    if not username:
        messages.success(request, "请登录后再访问！")
        return render(request, 'login.html')
    if request.method == 'GET':
        reminder_list = remind.objects.all()
        context = {"reminder_list": reminder_list}
        # return redirect('/index')
        return render(request, 'index.html', context=context)


def test(request):
    # 指定要访问的页面，render的功能：讲请求的页面结果提交给客户端
    if request.method == 'GET':
        return render(request, 'test.html')


# 注册页面
def register(request):
    #if request.method == 'GET':
        #return render(request, 'register.html')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        gender = request.POST.get('gender')
        tel = request.POST.get('tel')
        if username == '':
            messages.success(request, "用户名不能为空！")
            return render(request, 'login.html')
        elif password == '':
            messages.success(request, "密码不能为空！")
            return render(request, 'login.html')
        else:
            models.Users.objects.create(user_id=username, user_psd=password, user_gender=gender, user_tel=tel)
            messages.success(request, "注册成功！")
            return redirect('/login')


def logout(request):
    if request.method == 'GET':
        messages.success(request, "您已退出登录！")
        request.session.flush()
        return redirect('/login')

def profile_edit(request):
    if request.method == 'GET':
        username = request.session.get('username', None)
        u = Users.objects.get(user_id=username)
        context = {"u": u}
        return render(request, 'profile_edit.html', context=context)
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        gender = request.POST.get('gender')
        tel = request.POST.get('tel')
        confirmpassword = request.POST.get('confirm_password')
        if password == confirmpassword:
            models.Users.objects.filter(user_id=username).update(user_psd=password, user_gender=gender, user_tel=tel)
            messages.success(request, "修改成功！")
            username = request.session.get('username', None)
            u = Users.objects.get(user_id=username)
            context = {"u": u}
            return render(request, 'profile_edit.html', context=context)
            return render(request, 'profile_edit.html')
        else:
            messages.success(request, "两次密码输入不一致！")
            username = request.session.get('username', None)
            u = Users.objects.get(user_id=username)
            context = {"u": u}
            return render(request, 'profile_edit.html', context=context)
            return render(request, 'profile_edit.html')
