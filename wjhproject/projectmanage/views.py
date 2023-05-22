from django.utils import timezone
import pytz
import seaborn as sns
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.shortcuts import render, redirect
from pandas import DataFrame, read_csv
from sklearn.metrics import roc_auc_score
from datamanage.models import dataset
import os
from . import models
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from distributed.protocol.tests.test_torch import torch
from django.contrib import messages
from pyod.models.iforest import IForest
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import LabelEncoder
import warnings
# Create your views here.
from .models import project


def startproject(request):
    if request.method == 'GET':
        dataset_list = dataset.objects.all()
        context = {"dataset_list": dataset_list}
        return render(request, "startproject.html", context=context)


def runproject(request):
    if request.method == 'POST':
        username = request.session.get('username', None)
        pretraining_ratio = request.POST.get('pretraining_ratio')
        pretraining_ratio = float(pretraining_ratio)
        max_epoch = request.POST.get('epoch')
        max_epoch = int(max_epoch)
        batch_size = request.POST.get('batch_size')
        batch_size = int(batch_size)
        virtual_batch_size = request.POST.get('virtual_batch_size')
        virtual_batch_size = int(virtual_batch_size)
        datasetname = request.POST.get('datasetname')
        data = dataset.objects.get(dataset_name=datasetname)
        file_path = data.dataset_url
        tz = pytz.timezone('Asia/Shanghai')
        # 返回时间格式的字符串
        now_time = timezone.now().astimezone(tz=tz)
        now_time_str = now_time.strftime("%Y.%m.%d %H:%M:%S")
        projectname = datasetname + '_project' + now_time_str
        biliurl = bili(file_path)
        embedded_X, lossurl = pretrain(datasetname, pretraining_ratio, max_epoch, batch_size, virtual_batch_size,
                                       file_path)
        acc, recall, prec, auc, outputurl = detection(embedded_X)
        # detection(datasetname, pretraining_ratio, max_epoch, batch_size, virtual_batch_size, file_path)
        models.project.objects.create(project_name=projectname, loss_url=lossurl, bili_url=biliurl,
                                      output_url=outputurl, project_acc=acc, project_recall=recall,
                                      project_prec=prec, project_auc=auc, project_time=now_time_str, user_id=username)
        output_list = project.objects.get(project_name=projectname)
        # context = {"x": output_list, "batch_size": batch_size, "virtual_batch_size": virtual_batch_size, "max_epoch": max_epoch}
        #
        # data_set = read_csv(output_list.output_url, nrows=50)
        # data = data_set.values[:, :]
        # paginator = Paginator(data, 10)
        # # 获取当前的页码数，默认为1
        # page = request.GET.get("page", 1)
        # # 把当前的页码数转换为整数类型
        # currentPage = int(page)
        # try:
        #     test_data = paginator.page(currentPage)  # 获取当前页码的记录
        # except PageNotAnInteger:
        #     test_data = paginator.page(1)  # 如果用户输入的页码不是整数时,显示第1页的内容
        # except EmptyPage:
        #     test_data = paginator.page(paginator.num_pages)  # 如果用户输入的页码不是整数时,显示第1页的内容

        messages.success(request, "运行成功！")
        # return render(request, 'output.html', locals())
        return redirect('/projectlist')


def bili(file_path):
    df = pd.read_csv(file_path)
    colors = ["#0101DF", "#DF0101"]
    fig = sns.countplot('Class', data=df, palette=colors)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    bili_url = './projectmanage/images/bili.png'
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(bili_url, dpi=400)
    plt.savefig(bili_url)
    return bili_url


def pretrain(datasetname, pretraining_ratio, max_epoch, batch_size, virtual_batch_size, file_path):
    df = pd.read_csv(file_path,
                     usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                              26, 27, 28, 30])
    df.to_csv('./datamanage/output/creditcard.csv', index=False, header=False)
    train = df[1:]
    target = '1'

    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims = {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < 200:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

    unused_feat = ['Set']
    features = [col for col in train.columns if col not in unused_feat + [target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    X_test = np.genfromtxt('./datamanage/output/creditcard.csv', delimiter=',')
    df = pd.read_csv(file_path,
                     usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                              26, 27, 28, 30])
    df = df.sample(frac=1)

    # 正常数据492条
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]
    # 数据合并
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    # 随机取样做数据混洗
    new_df = normal_distributed_df.sample(frac=1, random_state=42)
    new_df.to_csv('./datamanage/output/X_train.csv', index=False, header=False)
    X_train = np.genfromtxt('./datamanage/output/X_train.csv', delimiter=',')
    # TabNetPretrainer
    unsupervised_model = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=40,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax'  # "sparsemax"
    )

    max_epochs = max_epoch \
        if not os.getenv("CI", False) else 2
    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_train],
        max_epochs=max_epochs, patience=3,
        batch_size=batch_size, virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=pretraining_ratio,
    )
    plt.plot(unsupervised_model.history['loss'])
    # plt.show()
    x = datasetname.split(".", 1)
    loss_saveurl = './projectmanage/images/' + x[0] + '.png'
    plt.savefig(loss_saveurl)
    reconstructed_X, embedded_X = unsupervised_model.predict(X_test)
    print('执行到这了')
    return embedded_X, loss_saveurl


def detection(embedded_X):
    model = IForest(n_estimators=100,
                    max_samples='auto', contamination=0.002,
                    max_features=1.0, bootstrap=False,
                    n_jobs=1, behaviour='old',
                    random_state=None, verbose=0)
    # n_estimators 集合中基估计量的数目。
    # max_samples 从X中抽取的样本数，用来训练每个基估计量。
    # contamination 数据集的污染量，即数据集中异常值的比例。
    # max_features 从X中提取的用于训练每个基估计量的特征数。
    # n_jobs 为拟合和预测并行运行的作业数
    # verbose 控制树构建过程的冗长程度。
    print('到这咯')
    model.fit(embedded_X)
    s = model.predict(embedded_X)
    b = [k for k in range(len(s)) if s[k] == 1]
    df = pd.read_csv('./datamanage/upload_files/creditcard.csv',
                     usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                              26, 27, 28, 30])
    df.to_csv('./datamanage/output/creditcard2.csv', header=False)
    for_output = np.genfromtxt('./datamanage/output/creditcard2.csv', delimiter=',')
    npoutput = np.empty(shape=(len(b), 30))
    for i in range(len(b)):
        npoutput[i] = for_output[b[i]]
    output_dataframe = DataFrame(npoutput)
    output_dataframe.to_csv('./datamanage/output/output.csv', index=False, header=False)
    output_url = './datamanage/output/output.csv'
    X_test = np.genfromtxt('./datamanage/output/creditcard.csv', delimiter=',')
    lista = []
    for i in range(len(X_test)):
        lista.append(int(X_test[i][28]))
    count3 = 0
    for i in range(len(X_test)):
        if s[i] == 1:
            count3 = count3 + 1
    # 检测出的异常点总数

    count = 0
    for i in range(len(X_test)):
        if lista[i] == 1:
            if s[i] == 1:
                count = count + 1
    # 分类正确的异常点

    count1 = 0
    for i in range(len(X_test)):
        if lista[i] == 1:
            count1 = count1 + 1
    ##样本中共有多少异常点

    count2 = 0
    for i in range(len(X_test)):
        if lista[i] == 0:
            if s[i] == 0:
                count2 = count2 + 1
    # print(count2)
    # 分类正确的正常点

    b = [k for k in range(len(s)) if s[k] == 1]
    # print(b)

    acc = (count + count2) / len(X_test)
    # 准确率

    recall = count / count1
    # 召回率

    prec = count / len(b)
    # 查准率

    auc = roc_auc_score(lista, s)
    # AUC指标
    return acc, recall, prec, auc, output_url


def projectlist(request):
    project_list = project.objects.all()
    context = {"project_list": project_list}
    return render(request, "projectlist.html", context=context)


def p_preview(request):
    id = request.GET.get('id')
    p = project.objects.get(project_name=id)
    lossurl = '.' + p.loss_url
    biliurl = '.' + p.bili_url
    data_set = read_csv(p.output_url, nrows=50)
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
    request.session['projectid'] = id
    return render(request, "project_preview.html", locals())


def p_turnpage(request):
    id = request.session.get('projectid')
    data = project.objects.get(project_name=id)
    data_set = read_csv(data.output_url, nrows=50)
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
    return render(request, "project_preview.html", locals())

# def detection(datasetname, pretraining_ratio, max_epoch, batch_size, virtual_batch_size, file_path):
#     df = pd.read_csv(file_path)
#     train = df
#     target = '1'
#     if "Set" not in train.columns:
#         train["Set"] = np.random.choice(["train", "valid", "test"], p=[.6, .0, .4], size=(train.shape[0],))
#
#     train_indices = train[train.Set == "train"].index
#     test_indices = train[train.Set == "test"].index
#
#     nunique = train.nunique()
#     types = train.dtypes
#
#     categorical_columns = []
#     categorical_dims = {}
#     for col in train.columns:
#         if types[col] == 'object' or nunique[col] < 200:
#             print(col, train[col].nunique())
#             l_enc = LabelEncoder()
#             train[col] = train[col].fillna("VV_likely")
#             train[col] = l_enc.fit_transform(train[col].values)
#             categorical_columns.append(col)
#             categorical_dims[col] = len(l_enc.classes_)
#         else:
#             train.fillna(train.loc[train_indices, col].mean(), inplace=True)
#
#     unused_feat = ['Set']
#     features = [col for col in train.columns if col not in unused_feat + [target]]
#     cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
#     cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
#
#     X_train = train[features].values[train_indices]
#     X_test = train[features].values[test_indices]
#
#     # TabNetPretrainer
#     unsupervised_model = TabNetPretrainer(
#         cat_idxs=cat_idxs,
#         cat_dims=cat_dims,
#         cat_emb_dim=3,
#         optimizer_fn=torch.optim.Adam,
#         optimizer_params=dict(lr=2e-2),
#         mask_type='entmax'  # "sparsemax"
#     )
#     max_epochs = max_epoch if not os.getenv("CI", False) else 2
#     unsupervised_model.fit(
#         X_train=X_train,
#         eval_set=[X_test],
#         max_epochs=max_epochs, patience=4,
#         batch_size=batch_size, virtual_batch_size=virtual_batch_size,
#         num_workers=0,
#         drop_last=False,
#         pretraining_ratio=pretraining_ratio,
#     )
#     plt.plot(unsupervised_model.history['loss'])
#     x = datasetname.split(".", 1)
#     img_saveurl = './projectmanage/images/' + x[0] + '.png'
#     plt.savefig(img_saveurl)
#     reconstructed_X, embedded_X = unsupervised_model.predict(X_test)
#     assert (reconstructed_X.shape == embedded_X.shape)
#     model = IForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False,
#                     n_jobs=1, behaviour='old', random_state=None, verbose=0)
#     model.fit(embedded_X)
