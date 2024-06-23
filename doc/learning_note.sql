# Prompt

3.1 Prompt策略文本问答中可以通过以下Prompt策略引导和约束模型生成符合预期的文本。
- 策略-打基础：用户无法一次性写出比较符合要求的提示词，先制定一个能够明确表达主题的提示词，保证最基础的提示词也有不错的文本生成效果。再由简至繁，逐步增加细节和说明。打好基础是后续提示词优化的前提，基础提示词生成效果差，优化只会事倍功半。
- 策略-做说明：对提示词进行细节的补充说明，比如生成文本的结构、输出的格式等，将想要的逻辑梳理表达出来，可能让生成效果更加符合预期。
- 策略-搭结构：提示词的结构需要尽可能得直观，不要将所有的内容放在一行输入，适当的换行将提示词的内容结构拆分体现出来。一个结构清晰的提示词输入，能够让模型更好地理解你的意图。
- 策略-做预设：由于训练数据和文本生成策略的原因，模型有时候会编造一些错误的输出，做好各种场景的预设可以有效防止模型说胡话。



# 文本摘要
3.1 Prompt策略介绍文本摘要中可以通过以下Prompt策略引导和约束模型生成符合预期的文本。
- 策略-打基础：用户无法一次性写出比较符合要求的提示词，先制定一个能够明确表达主题的提示词，保证最基础的提示词也有不错的文本生成效果。再由简至繁，逐步增加细节和说明。打好基础是后续提示词优化的前提，基础提示词生成效果差，优化只会事倍功半。
- 策略-设角色：设定模型在生成文本时应扮演的角色，为模型提供一个上下文场景，指定文本生成的角度。
- 策略-做说明：对提示词进行细节的补充说明，比如生成文本的结构、输出的格式等，将想要的逻辑梳理表达出来，可能让生成效果更加符合预期。



# 文本分类
3.1 Prompt策略文本分类中可以通过以下Prompt策略引导和约束模型生成符合预期的文本。
- 策略-做说明：对提示词进行细节的补充说明，比如生成文本的结构、输出的格式等，将想要的逻辑梳理表达出来，可能让生成效果更加符合预期。
- 策略-给提示：用户无法一次性写出比较符合要求的提示词，先制定一个能够明确表达主题的提示词，保证最基础的提示词也有不错的文本生成效果。再由简至繁，逐步增加细节和说明。打好基础是后续提示词优化的前提，基础提示词生成效果差，优化只会事倍功半。
- 策略-搭结构：提示词的结构需要尽可能得直观，不要将所有的内容放在一行输入，适当的换行将提示词的内容结构拆分体现出来。一个结构清晰的提示词输入，能够让模型更好地理解你的意图。


# 信息提取
3.1 Prompt策略信息提取中可以通过以下Prompt策略引导和约束模型生成符合预期的文本。
- 策略-打基础：用户无法一次性写出比较符合要求的提示词，先制定一个能够明确表达主题的提示词，保证最基础的提示词也有不错的文本生成效果。再由简至繁，逐步增加细节和说明。打好基础是后续提示词优化的前提，基础提示词生成效果差，优化只会事倍功半。
- 策略-设角色：设定模型在生成文本时应扮演的角色，为模型提供一个上下文场景，指定文本生成的角度。
- 策略-做说明：对提示词进行细节的补充说明，比如生成文本的结构、输出的格式等，将想要的逻辑梳理表达出来，可能让生成效果更加符合预期。


---水表识别


--mindsrope
实验手册在线问答实验报告华为云实验账号请使用实验账号登录，
确保实验顺利进行账号名:Sandbox-Voyager1557用户名:Sandbox-user密码:************基于MindSpore编写代码实现图像分类随着电子技术的迅速发展，
人们使用便携数码设备（如手机、相机等）获取花卉图像越来越方便，如何自动识别花卉种类受到了广泛的关注。由于花卉所处背景的复杂性，以及花卉自身的类间相似性和类内多样性，
利用传统的手工提取特征进行图像分类的方法，并不能很好地解决花卉图像分类这一问题。　　　本实验为基于卷积神经网络实现的花卉识别实验，与传统图像分类方法不同，卷积神经网络无需人工提取特征，
可以根据输入图像，自动学习包含丰富语义信息的特征，得到更为全面的花卉图像特征描述，可以很好地表达图像的不同类别信息。操作前提：登录华为云进入【实验操作桌面】，打开Chrome浏览器进入华为云登录页面。
选择【IAM用户登录】模式，于登录对话框中输入系统为您分配的华为云实验账号和密码登录华为云，如下图所示：注意：账号信息详见实验手册上方，切勿使用您自己的华为云账号登录。
1.准备数据1.1.创建OBS对象存储服务（Object Storage Service，OBS）是一个基于对象的海量存储服务，为客户提供海量、安全、高可靠、低成本的数据存储能力，包括：创建、修改、删除桶，上传、下载、删除对象等。)
点击“控制台”->“服务列表”-> 选择“存储”的“对象存储服务 OBS”，进入对象存储页面后，右上角点击“创建桶”。参数：①复制桶配置：不选②区域：华北-北京四③ 数据冗余存储策略：单AZ存储④桶名称：
自定义即可(需要记住此名称以备后续步骤使用)⑤ 存储类别：标准存储⑥桶策略：私有⑦ 默认加密：关闭⑧归档数据直读：关闭点击右下角“立即创建”，如下图所示：创建完成回到对象存储服务列表，点击刚刚创建的OBS桶名称进入详情页，
选择左侧“对象”-＞“新建文件夹”（存放后续步骤的数据文件）。文件夹名称：data；点击“确定”，
如下图：2. 准备开发环境2.1. 在“ModelArts控制台 > 开发环境 > Notebook”页面中，点击左上角“创建”按钮，进入创建Notebook页面。2.2 创建名称：自定义，如：notebook-flowers，运行时间选择1小时，
基于mindspore_1.9.0-cann_6.0.0-py_3.7-euler_2.8.3镜像,类型为ASCEND，规格选择Ascend: 1*Ascend910|ARM: 24核 96GB,然后点击“立即创建”如下图所示:点击“提交”：
最后点击“立即返回”2.3 填写完notebook相关配置信息之后，点击右下角“立即创建”按钮，确认产品规格，点击“提交”。返回Notebook页面等待notebook的状态为“运行中”后，单击操作栏的“打开”，
进入JupyterLab页面。JupyterLab操作请参见：JupyterLab简介及常用操作URL：https://support.huaweicloud.com/devtool-modelarts/devtool-modelarts_0013.html2.
4 打开JupyterLab的New--Notebook。2.5 选择Kelnel:Mindspore,如果选择成功，右上角会显示"Mindspore"3. 点击➕号，创建带代码行：3.创建算法工程3.1 导入实验环境：Glob模块主
要用于查找符合特定规则的文件路径名，类似使用windows下的文件搜索； os模块主要用于处理文件和目录，比如：获取当前目录下文件，删除制定文件，改变目录，查看文件大小等；MindSpore是目前业
界流行的深度学习框架之一，在图像，语音，文本，目标检测等领域都有深入的应用，也是该实验的核心，主要用于定义占位符，定义变量，创建卷积神经网络模型；numpy是一个基于python的科学计算包，
在该实验中主要用来处理数值运算。                        拷贝代码#easydict模块用于以属性的方式访问字典的值from easydict import EasyDict as edict#glob模块主要用于查找符合特
定规则的文件路径名，类似使用windows下的文件搜索import glob#os模块主要用于处理文件和目录import osimport numpy as npimport matplotlib.pyplot as pltimport mindspore#导入
mindspore框架数据集import mindspore.dataset as ds#vision.c_transforms模块是处理图像增强的高性能模块，用于数据增强图像数据改进训练模型。import mindspore.dataset.vision.c_

transforms as CV#c_transforms模块提供常用操作，包括OneHotOp和TypeCastimport mindspore.dataset.transforms.c_transforms as Cfrom mindspore.common import dtype as
 mstypefrom mindspore import context#导入模块用于初始化截断正态分布from mindspore.common.initializer import TruncatedNormalfrom mindspore import nnfrom mindspore.
 train import Modelfrom mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitorfrom mindspore.train.serialization import
 load_checkpoint, load_param_into_netfrom mindspore import Tensor# 设置MindSpore的执行模式和设备context.set_context(mode=context.GRAPH_MODE, device_target="Ascen
 d")3.2 定义变量                        拷贝代码cfg = edict({    'data_path': 'flower_photos',    'data_size':3670,    'image_width': 100,  # 图片宽度    'image_hei
 ght': 100,  # 图片高度    'batch_size': 32,    'channel': 3,  # 图片通道数    'num_class':5,  # 分类类别    'weight_decay': 0.01,    'lr':0.0001,  # 学习率    'drop
 out_ratio': 0.5,    'epoch_size': 400,  # 训练次数    'sigma':0.01,        'save_checkpoint_steps': 1,  # 多少步保存一次模型    'keep_checkpoint_max': 1,  # 最多保存
 多少个模型    'output_directory': './',  # 保存模型路径    'output_prefix': "checkpoint_classification"  # 保存模型文件名字})4. 数据集获取与预处理该数据集是开源数据集，总共包
 括5种花的类型：分别是daisy（雏菊，633张），dandelion（蒲公英，898张），roses（玫瑰，641张），sunflowers（向日葵，699张），tulips（郁金香，799张），保存在5个文件夹当中，总共3670张，大
 小大概在230M左右。为了在模型部署上线之后进行测试，数据集在这里分成了flower_train和flower_test两部分。数据读取并处理流程如下：1）	MindSpore的mindspore.dataset提供了ImageFolderData
 setV2函数，可以直接读取文件夹图片数据并映射文件夹名字为其标签(label)。这里我们使用ImageFolderDatasetV2函数	读取'daisy','dandelion','roses','sunflowers','tulips'数据。并将这五
 类标签映射为：	{'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4}2）	使用RandomCropDecodeResize、HWC2CHW、TypeCast、shuffle进行数据预处理4.1 获取数据集

                         拷贝代码# 解压数据集，只需要第一次运行时解压，第二次无需再解压!wget https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-le
                         arning/flower_photos.zip !unzip flower_photos.zip4.2 数据预处理                        拷贝代码#从目录中读取图像的源数据集。de_dataset = ds.
                         (cfg.data_path,                                   class_indexing={'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4})#解码前将输入图像裁剪成任意大小和宽高比。transform_img = CV.RandomCropDecodeResize([cfg.image_width,cfg.image_height], scale=(0.08, 1.0), ratio=(0.75, 1.333))  #改变尺寸#转换输入图像；形状（H, W, C）为形状（C, H, W）。hwc2chw_op = CV.HWC2CHW()#转换为给定MindSpore数据类型的Tensor操作。type_cast_op = C.TypeCast(mstype.float32)#将操作中的每个操作应用到此数据集。de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=8, operations=transform_img)de_dataset = de_dataset.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=8)de_dataset = de_dataset.map(input_columns="image", operations=type_cast_op, num_parallel_workers=8)de_dataset = de_dataset.shuffle(buffer_size=cfg.data_size)5 划分训练集与测试集1)	按照8:2的比列将数据划分为训练数据集和测试数据集2)	对训练数据和测试数据分批次（batch）                        拷贝代码#划分训练集测试集(de_train,de_test)=de_dataset.split([0.8,0.2])#设置每个批处理的行数#drop_remainder确定是否删除最后一个可能不完整的批（default=False）。#如果为True，并且如果可用于生成最后一个批的batch_size行小于batch_size行，则这些行将被删除，并且不会传播到子节点。de_train=de_train.batch(cfg.batch_size, drop_remainder=True)#重复此数据集计数次数。de_test=de_test.batch(cfg.batch_size, drop_remainder=True)print('训练数据集数量：',de_train.get_dataset_size()*cfg.batch_size)#get_dataset_size()获取批处理的大小。print('测试数据集数量：',de_test.get_dataset_size()*cfg.batch_size)data_next=de_dataset.create_dict_iterator(output_numpy=True).__next__()print('通道数/图像长/宽：', data_next['image'].shape)print('一张图像的标签样式：', data_next['label'])  # 一共5类，用0-4的数字表达类别。plt.figure()plt.imshow(data_next['image'][0,...])plt.colorbar()plt.grid(False)plt.show()输出结果：6. 构建CNN图像识别模型6.1  定义图像识别模型                        拷贝代码# 定义CNN图像识别网络class Identification_Net(nn.Cell):    def __init__(self, num_class=5,channel=3,dropout_ratio=0.5,trun_sigma=0.01):  # 一共分五类，图片通道数是3        super(Identification_Net, self).__init__()        self.num_class = num_class        self.channel = channel        self.dropout_ratio = dropout_ratio        #设置卷积层        self.conv1 = nn.Conv2d(self.channel, 32,                               kernel_size=5, stride=1, padding=0,                               has_bias=True, pad_mode="same",                               weight_init=TruncatedNormal(sigma=trun_sigma),bias_init='zeros')        #设置ReLU激活函数        self.relu = nn.ReLU()        #设置最大池化层        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2,pad_mode="valid")        self.conv2 = nn.Conv2d(32, 64,                               kernel_size=5, stride=1, padding=0,                               has_bias=True, pad_mode="same",                               weight_init=TruncatedNormal(sigma=trun_sigma),bias_init='zeros')        self.conv3 = nn.Conv2d(64, 128,         kernel_size=3, stride=1, padding=0,                               has_bias=True, pad_mode="same",                               weight_init=TruncatedNormal(sigma=trun_sigma),bias_init='zeros')        self.conv4 = nn.Conv2d(128, 128,                               kernel_size=3, stride=1, padding=0,                               has_bias=True, pad_mode="same",                               weight_init=TruncatedNormal(sigma=trun_sigma), bias_init='zeros')        self.flatten = nn.Flatten()        self.fc1 = nn.Dense(6*6*128, 1024,weight_init =TruncatedNormal(sigma=trun_sigma),bias_init = 0.1)        self.dropout = nn.Dropout(self.dropout_ratio)        self.fc2 = nn.Dense(1024, 512, weight_init=TruncatedNormal(sigma=trun_sigma), bias_init=0.1)        self.fc3 = nn.Dense(512, self.num_class, weight_init=TruncatedNormal(sigma=trun_sigma), bias_init=0.1)    #构建模型    def construct(self, x):        x = self.conv1(x)        #print(x.shape)        x = self.relu(x)        x = self.max_pool2d(x)        x = self.conv2(x)        x = self.relu(x)        x = self.max_pool2d(x)        x = self.conv3(x)        x = self.max_pool2d(x)        x = self.conv4(x)        x = self.max_pool2d(x)        x = self.flatten(x)        x = self.fc1(x)        x = self.relu(x)        #print(x.shape)        x = self.dropout(x)        x = self.fc2(x)        x = self.relu(x)        x = self.dropout(x)        x = self.fc3(x)        return x6.2 模型训练、测试、预测                        拷贝代码net=Identification_Net(num_class=cfg.num_class, channel=cfg.channel, dropout_ratio=cfg.dropout_ratio)#计算softmax交叉熵。net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")#optfc_weight_params = list(filter(lambda x: 'fc' in x.name and 'weight' in x.name, net.trainable_params()))other_params=list(filter(lambda x: 'fc' not in x.name or 'weight' not in x.name, net.trainable_params()))group_params = [{'params': fc_weight_params, 'weight_decay': cfg.weight_decay},                {'params': other_params},                {'order_params': net.trainable_params()}]#设置Adam优化器net_opt = nn.Adam(group_params, learning_rate=cfg.lr, weight_decay=0.0)#net_opt = nn.Adam(params=net.trainable_params(), learning_rate=cfg.lr, weight_decay=0.1)model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"acc"})loss_cb = LossMonitor(per_print_times=de_train.get_dataset_size()*10)config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,                             keep_checkpoint_max=cfg.keep_checkpoint_max)ckpoint_cb = ModelCheckpoint(prefix=cfg.output_prefix, directory=cfg.output_directory, config=config_ck)print("============== Starting Training ==============")model.train(cfg.epoch_size, de_train, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)# 使用测试集评估模型，打印总体准确率metric = model.eval(de_test)print(metric)输出结果：7. 图像分类模型验证7.1 加载训练模型                        拷贝代码#加载模型import osCKPT = os.path.join(cfg.output_directory,cfg.output_prefix+'-'+str(cfg.epoch_size)+'_'+str(de_train.get_dataset_size())+'.ckpt')net = Identification_Net(num_class=cfg.num_class, channel=cfg.channel, dropout_ratio=cfg.dropout_ratio)load_checkpoint(CKPT, net=net)model = Model(net)7.2 验证推理                        拷贝代码# 预测class_names = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}test_ = de_test.create_dict_iterator().__next__()test = Tensor(test_['image'], mindspore.float32)predictions = model.predict(test)predictions = predictions.asnumpy()true_label = test_['label'].asnumpy()#显示预测结果for i in range(10):    p_np = predictions[i, :]    pre_label = np.argmax(p_np)    print('第' + str(i) + '个sample预测结果：', class_names[pre_label], '   真实结果：', class_names[true_label[i]])输出结果：点击【Untitled.ipynb--Download】下载,然后打开浏览器点击“控制台”->“服务列表”-> 选择“存储”的“对象存储服务 OBS”，进入对象存储页面创建的文件夹data然后上传刚才下载的Untitled.ipynb文件，8.实验总结本章提供了一个基于华为MindSpore框架的图像识别实验。该实验演示了如何利用华为云ModelArts完成图像识别任务。本章对实验实验做了详尽的剖析。阐明了整个实验功能、结构与流程是如何设计的，详细解释了如何解析数据、如何构建深度学习模型、如何保存模型等内容。部署后的实验多个类别图片下进行测试，结果表明实验实验具有较快的推断速度和较好的识别性能。读者可以在该实验实验的基础上开发更有针对性的应用实验基于MindSpore编写代码实现图像分类结束实验延时00 : 38 : 23已完成100 %





