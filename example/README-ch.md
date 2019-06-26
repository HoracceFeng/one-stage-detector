      Editor   :   Haonan Feng
      Date     :   2019.05.20


这套代码旨在对 one-stage 检测器进行汇总。通过统一数据部署，模型训练，结果输出三部分的格式，为各位开发检测模型算法提供一套快捷的开发代码，同时希望能方便模型的后续部署工作。


## 这套代码的构造初衷是：
- 数据输入：使用 `voc` 格式部署数据，通过 `ImageSets/Main/train.txt` 控制模型训练需要使用的图片数据
- 模型设计：选择不同的 `backbone`, `head`, `loss`, 可以自由组建不同形式的模型
- 结果输出：输出结果部分有独立脚本 `detect.py` 方便后续模型部署的工作。另外，输出结果可以无缝对接到 [detection_evaluate](http://10.202.56.200:81/horacce/detection_evaluation) 项目中直接进行模型性能测试。 


## 代码结构参考:
- [keras-yolo3](https://github.com/experiencor/keras-yolo3)
- [pytorch_yolov3](https://github.com/ultralytics/yolov3)
- [ssds.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch)


## 目前代码支持：
- 无缝对接 darknet，可直接将源码训练得到的 yolov3 模型迁移过来继续开发
- pytorch yolov3 train, either train from scratch or train from darknet-pretrain-model


## Requirements:
- python3
- numpy
- opencv-python
- torch >= 1.0.0
- matplotlib
- tqdm


## TODO:
- [] ssd-series support
- [] tensorboard viz
- [] block-building structure 
- [] Probability-based Detector Module develop : "Probabilistic Object Detection: Definition and Evaluation" (arXiv:1811.10800v3)
- [] Light-weight RetinaNet Develop : "Light-Weight RetinaNet for Object Detection"



欢迎各位协助开发


## 使用方式
1. 正确配置 `cfg/config.json`
2. 模型训练： ```python train.py -c cfg/config.json```
3. (optional) 默认各部分损失函数的权重相等，但是若需要作进一步的训练，可使用权重更新，需要时间较长，慎用。  ```python train.py -c cfg/config.json -e True ```
4. 模型检测： ```python detect.py -c cfg/config.json```


## 代码修改和开发
为方便对该代码进行个性化的开发和使用，我在备注中加入了便于检索的`tag`
- ```grep -nr '###' [*.py]``` 可以输出具体某个 `*py` 代码的目录
- ```grep -nr '#\*#' [*.py]``` 可以输出具体该 `*py` 代码中可修改优化的部分


## 配置文件详解
```
{
    ### 模型参数
    "model":{
        ### 使用 darknet 迁移过来的模型，请配置 “darknet”
        "darknet":{
            "darknet_cfg":    "tmp/yolo3_model/FT-Prohibit-yolov3-tiny-anchor5-tt100k-512-test.cfg",
            "head":           "yolov3"                                ## 目前只支持 yolov3
        },

        ### 开发中，不可用，后续支持 ssd
        "experiment":{
            "backcbone":      "yolov3-tiny",
            "head":           "yolov3",
            “loss”:           "yolo"
        },

        "weights":{
            "format":         "darknet",                              ## 无需修改 
            "pretrain":       "/code/weights/yolov3-tiny.weights",    ## 预训练模型路径
            "resume":         "False",                                ## 控制是否使用预训练模型
            "transfer":       ,                                       ## 数字可指定预训练模型使用到第几层（置空为全用）
            "freeze":         "False"                                 ## 控制是否训练 backbone
        },

        "parameters":{
            "dictfile":       "/code/data/sign/FT-prohibit.names",
            "autoevolve":     "False",                                ## 开发中，自动更新参数
            "notest":         "False",                                ## 训练时是否每个 epoch 进行测试
            "nosave":         "False",                                ## 训练时是否保存权重
            "save_interval":  1,                                      ## 间隔多少 epoch 保存权重
            "debug":          "False"
        }

    },

    ### 模型训练使用的参数，仅用于 `train.py`
    "train":{
        
        "name":               "TC-sign-pytorch-yolov3",               ## 项目名称，用于自动生成结果保存的文件夹，以及权重名称前缀
       
        ### 数据输入格式
        "data":{
            "format":         "voc",                                  ## 数据输入格式，"voc"即支持voc格式；"darknet"即需要使用 darknet 源码转化后的 labels 文件，
            "txtpath":        "/data/sign_detect/mini/ImageSets/Main/train-fullpath.txt",
            "augment":        "True"                                  ## 是否进行数据增广
        },

        "parameters":{
            "max_epoch":      100,                                    ## 最大训练 epoch 数
            "batch":          16,                                     ## batch_size
            "size":           512,                                    ## 输入数据size，仅支持正方形。非正方形会被 resize+pad，要支持长方形需更改 `data/dataset.py`
            "learning_rate":  0.001,                                  ## 初始学习率，默认使用 inverse-exp 进行学习率更新
            "lr_scheduler":   "",                                     ## 未支持。grep -nr '### #\*# SCHEDULER' 手动修改学习率变化
            "momentum":       0.9127,
            "weight_decay":   0.0004841,
            "gpus":           ""                                      ## 支持多gpu
        }

    },

    ### 模型测试使用的参数，仅用于 `train.py`
    "test":{

        "data":{
            "format":         "darknet",                              ## 数据输入格式，"voc"即支持voc格式；"darknet"即需要使用 darknet 源码转化后的 labels 文件
            "txtpath":        "/data/sign_detect/mini/ImageSets/Main/train-fullpath.txt"
        },

        "parameters":{
            "start_epoch":    1,                                      ## 第几个 epoch 开始测试
            "interval":       1,                                      ## 测试间隔 epoch
            "batch":          2,                                      ## 测试 batch_sizer
            "size":           512                                     ## 测试图片size。要支持长方形需更改 `data/dataset.py`                         
        }
    },


    ### 仅用于 `detect.py` 
    "evaluate":{

        "model":{
            "weights_path":   "/code/weights/TC-sign-pytorch-yolov3/best.pt",     ## 检测使用的权重，支持 .pt 或者 .weight
            "conf_thresh":    0.5,
            "nms_thresh":     0.5
        },

        "data":{
            "name":           "TC-test",                                          ## 检测项目名称
            "dirpath":        "/data/sign_detect/mini/JPEGImages/test",           ## 检测图片路径
            "isimages":       "True"                                              ## 检测文件为图片时，不用修改
        },

        "parameters":{
            "outdir":         "/code/output",                                     ## 结果输出路径
            "size":           512,                                                ## 检测图片 size，仅支持正方性，非正方形会被 resize+pad
            "save_images":    "True",                                             ## 是否保存结果图片
            "gpus":           ""                                                  ## 支持多gpu
        }

    }
}
```








