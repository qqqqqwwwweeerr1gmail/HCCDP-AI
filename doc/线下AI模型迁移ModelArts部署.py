
































































































































1 环境准备进入控制台页面后，点击左上角服务列表按钮,下拉找到【人工智能】,再找到【ModelArts】,点击进入 “ModelArts”控制台页面,如下图所示:进入ModelArts控制管理台，在左侧导航栏点击【开发环境】-> 【Notebook】，进入notebook列表页面，如下图所示：点击页面左上角“创建”按钮，新建一个notebook，填写参数，下图所示：说明：如GPU：1\*P100(16GB)|CPU:8核64GB规格售罄，请选择GPU：1\*V100(32GB)|CPU:8核64GB规格




点击“立即创建”，确认产品规格后，点击提交，完成Notebook的创建。返回Notebook列表页面，等待新创建Notebook状态变为“运行中”后，点击“打开”进入Notebook。进入Notebook页面后，打开terminal，如下图所示：输入如下命令，查看已安装Python环境信息                        拷贝代码conda info -e在右侧浏览器复制并打开此链接：https://devcloud.cn-north-4.huaweicloud.com/codehub/project/3674262af83841b49a35edcdcd2ac6d2/codehub/2136282/home?ref=main进入DINO代码仓库，下面将以此开源算法为例，演示如何在华为云Notebook上快速运行,算法详细介绍请参考 README.md 。① 在terminal里继续输入如下命令，克隆仓库                        拷贝代码git clone https://codehub.devcloud.cn-north-4.huaweicloud.com/DINO00002/DINO.gitcd DINO如上图所示，表示已完成代码克隆，点击左侧任务栏顶部刷新按钮，即可查看代码。② 查看Pytorch版本                        拷贝代码pip list | grep torch③ 安装其他需要的包                        拷贝代码pip install -r requirements.txt④ 编译CUDA算子                        拷贝代码cd models/dino/ops                        拷贝代码python setup.py build install                        拷贝代码# 单元测试（应查看所有检查均为True）python test.py                        拷贝代码cd ../../..  # 回到代码主目录


2 数据准备2.1 创建桶进入控制台，将光标移动至左边栏，弹出菜单中选择“服务列表”->“存储”->“对象存储服务OBS”，如下图所示：点击“创建桶”按钮进入创建界面。开始创建。配置参数如下：① 复制桶配置：不选② 区域：华北-北京四③ 桶名称：自定义，将在后续步骤使用④ 数据冗余存储策略：单AZ存储⑤ 默认存储类别：标准存储⑥ 桶策略：私有⑦ 默认加密：关闭⑧ 归档数据直读：关闭单击“立即创建”>“确定”，完成桶创建。点击创建的“桶名称”->“对象”->“新建文件夹”，创建一个文件夹，命名为coco2017_subset，用于存放后续数据集。2.2 数据集下载复制并打开此链接：https://developer.huaweicloud.com/develop/aigallery/dataset/detail?id=6d7347f8-f674-4900-a5fd-d6adb158cca7下载COCO 2017数据集子集 COCO2017_subset100。                        拷贝代码COCO2017_subset100/  ├── train2017/  ├── val2017/  └── annotations/  	├── instances_train2017.json  	└── instances_val2017.json该数据集包括train（100张），val（100张）及标注文件。进入COCO2017_subset100数据集页面，点击“下载”，选择云服务区域为“华北-北京四”，点击确认。进入下载详情页面后，下载方式选择对象存储服务（OBS），目标区域选择华北-北京四，目标路径选择1中在OBS中创建的路径，用于数据集存储，如下图所示：点击“确认”，跳转至我的下载页面，可以查看数据集下载详情，等待数据集下载完成，如下图所示：返回Notebook页面，新建一个ipynb文件，补充以下代码，并运行，导入COCO2017_subset100数据集，运行完毕后，点击任务栏上方“刷新”按钮，即可查看导入dataset目录，如下图所示：                        拷贝代码import moxing as moxmox.file.copy_parallel(${obs_path},${notebook_path})

说明：${obs_path}为OBS存储数据集的位置${notebook_path}为数据集在notebook中的存储路径./coco2017_subset，与DINO在同级目录


2.3 模型下载打开terminal,输入如下命令，下载DINO model checkpoint “checkpoint0011_4scale.pth”。下载完成后，点击左侧刷新按钮，即可查看文件夹ckpts，用于存放下载的checkpoint。                        拷贝代码# 待换链接wget -P ckpts https://sandbox-expriment-files.obs.cn-north-1.myhuaweicloud.com:443/20221228/checkpoint0011_4scale.pth3 推理及可视化打开DINO目录下的inference_and_visualization.ipynb，选择Kernel Pytorch-1.8，如下图所示：按步骤运行代码查看推理结果。

4 模型部署准备在跑通DINO的推理及可视化部分后，在ModelArts服务中可以将AI模型创建为AI应用，然后将AI应用快速部署为在线推理服务，流程图如下所示：◉ 开发模型：模型开发可以在ModelArts服务中进行，也可以在您的本地开发环境进行。◉ 创建AI应用：把模型文件、配置文件和推理文件导入到ModelArts的模型仓库中，进行版本化管理，并构建为可运行的AI应用。◉ 部署服务：把AI应用在资源池中部署为容器实例，注册外部可访问的推理API。◉ 推理：在您的应用中增加对推理API的调用，在业务流程中集成AI推理能力。4.1 编写推理代码在Notebook-DINO路径下，创建一个名称为customize_service的py文件，参考inference_and_visualization.ipynb编写推理代码,可参照模型推理代码编写说明：https://support.huaweicloud.com/inference-modelarts/inference-modelarts-0057.html

4.1.1 导包1) 在模型代码推理文件“customize_service.py”中，需要添加一个子类，该子类继承对应模型PyTorch类型的父类PTServingBaseService,导入语句如下所示:


2) 参考inference_and_visualization.ipynb，将推理所需包导入到customize_service.py中                        拷贝代码import osimport torchimport jsonfrom io import BytesIOfrom collections import OrderedDictfrom PIL import Imagefrom util.slconfig import SLConfigfrom main import build_model_mainfrom util import box_opsimport datasets.transforms as T4.1.2 重写方法1) 重写初始化方法此方法用于初始化和加载预训练模型。首先需要构建一个类，这个类应当继承PTServingBaseService，且需定义模型配置文件路径及模型路径，并加载配置文件中的参数信息及coco名称等操作。                        拷贝代码class CustomizeService(PTServingBaseService):    def __init__(self, model_name, model_path):        root = os.path.dirname(os.path.abspath(__file__))        model_config_path = os.path.join(root, "config/DINO/DINO_4scale.py")        model_checkpoint_path = os.path.join(root, "ckpts/checkpoint0011_4scale.pth")        self.model_config_path = model_config_path        self.model_checkpoint_path = model_checkpoint_path        args = SLConfig.fromfile(self.model_config_path)        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'        self.model, criterion, self.postprocessors = build_model_main(args)        checkpoint = torch.load(self.model_checkpoint_path, map_location='cpu')        self.model.load_state_dict(checkpoint['model'])        _ = self.model.eval()        # load coco names        jsonfile = os.path.join(root,"util/coco_id2name.json")        with open(jsonfile) as f:            id2name = json.load(f)            self.id2name = {int(k): v for k, v in id2name.items()}        self.transform = T.Compose([            T.RandomResize([800], max_size=1333),            T.ToTensor(),            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        ])        self.resize_scale = 0.0        self.image = None2) 重写预处理方法此方法对自定义图像进行预处理，实际推理请求方法和后处理方法中的接口传入“data”当前支持两种content-type，即“multipart/form-data”和“application/json”。此处定义传入参数data为multipart/form-data类型。首先加载图片，将其转换为RGB格式的图片，接着对图像进行预处理，将PIL图像随机调整为形状列表中的目标大小,将PIL图像转换为tensor,最后用平均值和标准偏差归一化张量图像。


线下AI模型迁移ModelArts部署
通过云上快速调试，编写推理代码，保存自定义镜像方式进行AI应用的创建，并部署成华为云在线服务，方便调用。
操作前提：登录华为云
进入【实验操作桌面】，打开Chrome浏览器，选择“IAM 用户登录”，并在对话框中输入系统为您分配的华为云实验账号和密码进行登录。


注意：请使用实验手册上方账号信息，切勿使用您自己的华为云账号登录。


1 环境准备
进入控制台页面后，点击左上角服务列表按钮,下拉找到【人工智能】,再找到【ModelArts】,点击进入 “ModelArts”控制台页面,如下图所示:


进入ModelArts控制管理台，在左侧导航栏点击【开发环境】-> 【Notebook】，进入notebook列表页面，如下图所示：


点击页面左上角“创建”按钮，新建一个notebook，填写参数，下图所示：

说明：如GPU：1\*P100(16GB)|CPU:8核64GB规格售罄，请选择GPU：1\*V100(32GB)|CPU:8核64GB规格


点击“立即创建”，确认产品规格后，点击提交，完成Notebook的创建。

返回Notebook列表页面，等待新创建Notebook状态变为“运行中”后，点击“打开”进入Notebook。

进入Notebook页面后，打开terminal，如下图所示：


输入如下命令，查看已安装Python环境信息

                        拷贝代码
conda info -e
在右侧浏览器复制并打开此链接：https://devcloud.cn-north-4.huaweicloud.com/codehub/project/3674262af83841b49a35edcdcd2ac6d2/codehub/2136282/home?ref=main

进入DINO代码仓库，下面将以此开源算法为例，演示如何在华为云Notebook上快速运行,算法详细介绍请参考 README.md 。

① 在terminal里继续输入如下命令，克隆仓库

                        拷贝代码
git clone https://codehub.devcloud.cn-north-4.huaweicloud.com/DINO00002/DINO.git
cd DINO

如上图所示，表示已完成代码克隆，点击左侧任务栏顶部刷新按钮，即可查看代码。

② 查看Pytorch版本

                        拷贝代码
pip list | grep torch
③ 安装其他需要的包

                        拷贝代码
pip install -r requirements.txt
④ 编译CUDA算子

                        拷贝代码
cd models/dino/ops
                        拷贝代码
python setup.py build install
                        拷贝代码
# 单元测试（应查看所有检查均为True）
python test.py
                        拷贝代码
cd ../../..  # 回到代码主目录

2 数据准备
2.1 创建桶
进入控制台，将光标移动至左边栏，弹出菜单中选择“服务列表”->“存储”->“对象存储服务OBS”，如下图所示：


点击“创建桶”按钮进入创建界面。


开始创建。配置参数如下：

① 复制桶配置：不选

② 区域：华北-北京四

③ 桶名称：自定义，将在后续步骤使用

④ 数据冗余存储策略：单AZ存储

⑤ 默认存储类别：标准存储

⑥ 桶策略：私有

⑦ 默认加密：关闭

⑧ 归档数据直读：关闭

单击“立即创建”>“确定”，完成桶创建。

点击创建的“桶名称”->“对象”->“新建文件夹”，创建一个文件夹，命名为coco2017_subset，用于存放后续数据集。


2.2 数据集下载
复制并打开此链接：https://developer.huaweicloud.com/develop/aigallery/dataset/detail?id=6d7347f8-f674-4900-a5fd-d6adb158cca7

下载COCO 2017数据集子集 COCO2017_subset100。

                        拷贝代码
COCO2017_subset100/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
该数据集包括train（100张），val（100张）及标注文件。进入COCO2017_subset100数据集页面，点击“下载”，选择云服务区域为“华北-北京四”，点击确认。


进入下载详情页面后，下载方式选择对象存储服务（OBS），目标区域选择华北-北京四，目标路径选择1中在OBS中创建的路径，用于数据集存储，如下图所示：


点击“确认”，跳转至我的下载页面，可以查看数据集下载详情，等待数据集下载完成，如下图所示：


返回Notebook页面，新建一个ipynb文件，补充以下代码，并运行，导入COCO2017_subset100数据集，运行完毕后，点击任务栏上方“刷新”按钮，即可查看导入dataset目录，如下图所示：

                        拷贝代码
import moxing as mox
mox.file.copy_parallel(${obs_path},${notebook_path})
说明：

${obs_path}为OBS存储数据集的位置

${notebook_path}为数据集在notebook中的存储路径./coco2017_subset，与DINO在同级目录


2.3 模型下载
打开terminal,输入如下命令，下载DINO model checkpoint “checkpoint0011_4scale.pth”。下载完成后，点击左侧刷新按钮，即可查看文件夹ckpts，用于存放下载的checkpoint。

                        拷贝代码
# 待换链接
wget -P ckpts https://sandbox-expriment-files.obs.cn-north-1.myhuaweicloud.com:443/20221228/checkpoint0011_4scale.pth

3 推理及可视化
打开DINO目录下的inference_and_visualization.ipynb，选择Kernel Pytorch-1.8，如下图所示：


按步骤运行代码查看推理结果。


4 模型部署准备
在跑通DINO的推理及可视化部分后，在ModelArts服务中可以将AI模型创建为AI应用，然后将AI应用快速部署为在线推理服务，流程图如下所示：


◉ 开发模型：模型开发可以在ModelArts服务中进行，也可以在您的本地开发环境进行。

◉ 创建AI应用：把模型文件、配置文件和推理文件导入到ModelArts的模型仓库中，进行版本化管理，并构建为可运行的AI应用。

◉ 部署服务：把AI应用在资源池中部署为容器实例，注册外部可访问的推理API。

◉ 推理：在您的应用中增加对推理API的调用，在业务流程中集成AI推理能力。

4.1 编写推理代码
在Notebook-DINO路径下，创建一个名称为customize_service的py文件，参考inference_and_visualization.ipynb编写推理代码,可参照模型推理代码编写说明：https://support.huaweicloud.com/inference-modelarts/inference-modelarts-0057.html


4.1.1 导包

1) 在模型代码推理文件“customize_service.py”中，需要添加一个子类，该子类继承对应模型PyTorch类型的父类PTServingBaseService,导入语句如下所示:

                        拷贝代码
from model_service.pytorch_model_service import PTServingBaseService
2) 参考inference_and_visualization.ipynb，将推理所需包导入到customize_service.py中

                        拷贝代码
import os
import torch
import json
from io import BytesIO
from collections import OrderedDict
from PIL import Image
from util.slconfig import SLConfig
from main import build_model_main
from util import box_ops
import datasets.transforms as T
4.1.2 重写方法


1) 重写初始化方法

此方法用于初始化和加载预训练模型。首先需要构建一个类，这个类应当继承PTServingBaseService，且需定义模型配置文件路径及模型路径，并加载配置文件中的参数信息及coco名称等操作。

                        拷贝代码
class CustomizeService(PTServingBaseService):
    def __init__(self, model_name, model_path):

        root = os.path.dirname(os.path.abspath(__file__))
        model_config_path = os.path.join(root, "config/DINO/DINO_4scale.py")
        model_checkpoint_path = os.path.join(root, "ckpts/checkpoint0011_4scale.pth")
        self.model_config_path = model_config_path
        self.model_checkpoint_path = model_checkpoint_path

        args = SLConfig.fromfile(self.model_config_path)
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model, criterion, self.postprocessors = build_model_main(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        _ = self.model.eval()

        # load coco names
        jsonfile = os.path.join(root,"util/coco_id2name.json")
        with open(jsonfile) as f:
            id2name = json.load(f)
            self.id2name = {int(k): v for k, v in id2name.items()}
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.resize_scale = 0.0
        self.image = None
2) 重写预处理方法

此方法对自定义图像进行预处理，实际推理请求方法和后处理方法中的接口传入“data”当前支持两种content-type，即“multipart/form-data”和“application/json”。此处定义传入参数data为multipart/form-data类型。

首先加载图片，将其转换为RGB格式的图片，接着对图像进行预处理，将PIL图像随机调整为形状列表中的目标大小,将PIL图像转换为tensor,最后用平均值和标准偏差归一化张量图像。

                        拷贝代码
def _preprocess(self, data):
    preprocessed_data = {}
    for k, v in data.items():  # k的取值固定为'images'
        for file_name, file_content in v.items():
            image = Image.open(file_content).convert("RGB")
            long_side = max(image.size)
            # transform images
            self.image, _ = self.transform(image, None)
            self.resize_scale = max(self.image.size()) / float(long_side)
            preprocessed_data[k] = self.image
    return preprocessed_data
3) 重写推理方法

传入处理过的图片数据进入模型，return得分，标签及位置信息。



4) 重写后处理方法此方法对推理输出的数据进行后处理操作，设置阈值对得分及边框进行筛选，去掉多余的boxes，并对坐标信息进行处理。推理结果以“JSON”体的形式返回，detection_classes是每个检测框的标签，detection_boxes是每个检测框的四点坐标（y_min,x_min,y_max,x_max），detection_scores是每个检测框的置信度。                        拷贝代码def _postprocess(self, data):    thershold = 0.3  # set a thershold    scores = data['scores']    labels = data['labels']    boxes = box_ops.box_xyxy_to_cxcywh(data['boxes'])    select_mask = scores > thershold    box_label = [self.id2name[int(item)] for item in labels[select_mask]]    pred_dict = {        'boxes': boxes[select_mask],        'size': torch.Tensor([self.image.shape[1], self.image.shape[2]]),        'box_label': box_label    }    H, W = pred_dict['size'].tolist()    polygons = []    for box in pred_dict['boxes'].cpu():        unnormbbox = box * torch.Tensor([W, H, W, H])        unnormbbox[:2] -= unnormbbox[2:] / 2        unnormbbox /= self.resize_scale        [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()        polygons.append([int(bbox_y), int(bbox_x), int(bbox_y + bbox_h), int(bbox_x + bbox_w)])  # 坐标顺序[ y1, x1, y2, x2]        result = OrderedDict()    result['detection_classes'] = box_label    result['detection_scores'] = [round(v, 4) for v in scores[select_mask].tolist()]    result['detection_boxes'] = polygons        return result编写完推理脚本后，点CRTL+S进行保存。
# 4.2 模型打包在2.2中创建的ipynb文件中，新增一个cell，复制以下代码，将推理代码及模型文件打包到指定目录下。


4.3 本地调用打开terminal，输入以下命令，切换到infer路径下，启动本地推理代码


此时点击右上角新建按钮，新开一个terminal在新建的terminal中输入以下命令，通过传入本地图片调用测试推理代码进行推理验证。                        拷贝代码curl -kv -F 'images=@/home/ma-user/infer/model/1/figs/idea.jpg' -X POST http://127.0.0.1:8080/推理结果如下，返回图像的标签，尺寸以及边框的位置信息。

4.4 保存镜像步骤1中通过预置的镜像创建Notebook实例，在基础镜像上安装对应的自定义软件和依赖，并在管理页面上进行操作，进而完成将运行的实例环境以容器镜像的方式保存下来。保存的镜像中，安装的依赖包不会丢失，但持久化存储的部分（home/ma-user/work目录的内容）不会保存在最终产生的容器镜像中。须知1.复制以下链接：https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dev-container返回Notebook列表。或在ModelArts左侧菜单栏中选择“开发环境 > Notebook”，进入新版Notebook管理页面。2.在Notebook列表中，对于要保存的Notebook实例，单击右侧“操作”列中的“更多 > 保存镜像”，进入“保存镜像”对话框。3.在保存镜像对话框中，设置组织、镜像名称、镜像版本和描述信息。单击“确认”保存镜像。在“组织”下拉框中选择一个组织。如果没有组织，可以单击右侧的“立即创建”，创建一个组织。创建组织的详细操作请参见——创建组织：https://support.huaweicloud.com/usermanual-swr/swr_01_0014.html#section0同一个组织内的用户可以共享使用该组织内的所有镜像。4.镜像会以快照的形式保存，保存过程约5分钟，请耐心等待。此时不可再操作实例（对于打开的JupyterLab界面和本地IDE 仍可操作）。



4.镜像会以快照的形式保存，保存过程约5分钟，请耐心等待。此时不可再操作实例（对于打开的JupyterLab界面和本地IDE 仍可操作）。


须知：快照中耗费的时间仍占用实例的总运行时长，若在快照中时，实例因运行时间到期停止，将导致镜像保存失败。

5.镜像保存成功后，实例状态变为“运行中”，用户可在“镜像管理”页面查看到该镜像详情。

6.单击镜像的名称，进入镜像详情页，可以查看镜像版本/ID，状态，资源类型，镜像大小，SWR地址等。


5 创建AI应用并部署成在线服务点击左侧“AI应用管理 >AI应用 >我的AI应用”，点击“创建”配置参数参考下图：名称：自定义，如model-dino版本：默认元模型来源：从容器镜像中选择容器镜像所在的路径：点击右侧文件图标，选择保存的镜像及版本容器调用接口：HTTPS，端口号8443部署类型：在线服务启动命令：sh /home/ma-user/infer/run.shapis定义：开启，并在下方编辑apis,编辑完成后，点击保存，完成apis定义。apis定义为json数组格式，应包含以下字段：













































































































