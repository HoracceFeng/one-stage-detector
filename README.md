      Editor   :   Haonan Feng
      Date     :   2019.05.20


# Pipeline of One-Stage Detector 

This Project is a summary of SOTA one-stage detector, kind of like mm-detection, but I am trying to make this more convenient to use. The other reason to start this project is, the object detection algorithm is actually have a clear pipeline, but different repos in github usually have huge  difference in using their code. To figure this, I then decide to do this pipeline code and try unify them in just one strcture. I sware, this code is definitely a newbie-friendly and developer-friendly code. Just simply plugin and run, and you get the world.


One more thing, some code is basically forked and re-implemented by three projects listed below, thanks for their contributions. You can also follow their work by clicking the link in 'Reference'

For Chinese Version Introduction, [click here](https://github.com/HoracceFeng/one-stage-detector/tree/master/example/)


## Structure

- Input Loader: `Pascal VOC` structure are used in this code. I prefer to use a txt file in `ImageSets/Main/train.txt` to control the data I use in training

- Model Design: By choosing differents `backbons`, `head` and `loss` in config, you can simply your unique model and have some fun.

- Output Result: using `detect.py` with your trained model to get your results. It is also very convenient to use when you just want to build up an app. If you want to further analysis your results and get the evaluation report, just follow this repo [detection_evaluate](https://github.com/HoracceFeng/detection_eval) directly. 


## Code Catalog

For people who want to further study the algorithms by code or just want to do further development, here is a trick that may help you.

I add comments as `tag` for convenient search. If you use Linux, you can:
- ```grep -nr '###' [*.py]``` can output the catalog of `*.py` script
- ```grep -nr '#\*#' [*.py]``` can output which part in `*.py` script can be modified. [especially for trainee]


## Reference

- [keras-yolo3](https://github.com/experiencor/keras-yolo3)
- [pytorch_yolov3](https://github.com/ultralytics/yolov3)
- [ssds.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch)
- [detection_evaluate](https://github.com/HoracceFeng/detection_eval)


## Requirements:
- python3
- numpy
- opencv-python
- torch >= 1.0.0
- matplotlib
- tqdm


## Now Support：

- Yolov3, Yolov3-Tiny
  - model trained from *darknet* source code can transfer to this code directly with the origin `.cfg`, `.data` and `.name` file. 
  - directly trained by this code, either train from scratch or train from darknet-pretrain-model


## TODO:

[] ssd-series support
[] tensorboard viz
[] block-building structure 
[] Probability-based Detector Module develop : "Probabilistic Object Detection: Definition and Evaluation" (arXiv:1811.10800v3)
[] Light-weight RetinaNet Develop : "Light-Weight RetinaNet for Object Detection"



## How to Use

0. You should firstly install requirements listed above by just simply using `pip install [package]`
1. write `cfg/config.json` correctly. Detailed arguments listed below. Also, you can just use the example and modify the datapath to start training yolo3-tiny model directly.
2. train model： ```python train.py -c cfg/config.json```
3. (optional) The default weights for different parts of LOSS function is the same. But if you want to search a better weights for better result, you can try this ```python train.py -c cfg/config.json -e True ```. However, it may takes a very long time. So you can just skip it.
4. output result： ```python detect.py -c cfg/config.json```





## Detail of Configuration
```
{
    ### parameters of model
    "model":{
        ### if you use model trained by darknet, please make changes in “darknet”
        "darknet":{
            "darknet_cfg":    "tmp/yolo3_model/FT-Prohibit-yolov3-tiny-anchor5-tt100k-512-test.cfg",
            "head":           "yolov3"                                ## now only support yolov3
        },

        ### still under develop, will support SSD-series further
        "experiment":{
            "backcbone":      "yolov3-tiny",
            "head":           "yolov3",
            “loss”:           "yolo"
        },

        "weights":{
            "format":         "darknet",                              ## no need to train
            "pretrain":       "/code/weights/yolov3-tiny.weights",    ## pretrain model path
            "resume":         "False",                                ## use pretrain model or not
            "transfer":       ,                                       ## fill nothing means use the whole model, fill a number such as '3' means freeze the model before layer 3
            "freeze":         "False"                                 ## train the backbone or not [a little bit the same as `transfer`]
        },

        "parameters":{
            "dictfile":       "/code/data/sign/FT-prohibit.names",
            "autoevolve":     "False",                                ## under develop, in order to evlove the loss weight automatically
            "notest":         "False",                                ## if True, the model will be tested after each epoch
            "nosave":         "False",                                ## if True, the model will be saved after each epoch
            "save_interval":  1,                                      ## skip how many epoch to save the model 
            "debug":          "False"
        }

    },

    ### parameters to control model training，only use in `train.py`
    "train":{
        
        "name":               "TC-sign-pytorch-yolov3",               ## project name, will create a file with this name automatically, the saved model will use this name as prefix
       
        ### Data Input Loader
        "data":{
            "format":         "voc",                                  ## data input format, "voc" or "darknet". "voc" for "Pascal voc" dataset strucutre. "darknet" for `labels` directory use in darkent source code 
            "txtpath":        "/data/ImageSets/Main/train.txt",       ## a txtfile path that control which images to train
            "augment":        "True"                                  ## data augmentation or not
        },

        "parameters":{
            "max_epoch":      100,                                    ## max number of epoch to train
            "batch":          16,                                     ## batch_size
            "size":           512,                                    ## input size，only support square size. for rectangle data, just modify `dat/dataset.py` by following the comments.
            "learning_rate":  0.001,                                  ## learning rate. default update method is `inverse-exp`
            "lr_scheduler":   "",                                     ## under develop, not support yet
            "momentum":       0.9127,
            "weight_decay":   0.0004841,
            "gpus":           ""                                      ## support multi-gpu
        }

    },

    ### parameters to control the evaluation in training, only use in `train.py`
    "test":{

        "data":{
            "format":         "darknet",                              ## data input format, "voc" or "darknet". "voc" for "Pascal voc" dataset strucutre. "darknet" for `labels` directory use in darkent source code 
            "txtpath":        "/data/ImageSets/Main/eval.txt"         ## a txtfile path that control which images to evaluate
        },

        "parameters":{
            "start_epoch":    1,                                      ## test from which epoch
            "interval":       1,                                      ## test epoch intervel number
            "batch":          2,                                      ## test batch size, can set the same number as train, or a little bit larger
            "size":           512                                     ## input size of evaluation. for rectangle data, just modify `dat/dataset.py` by following the comments.dataset.py`                         
        }
    },


    ### parameters to control result output, only use in `detect.py` 
    "evaluate":{

        "model":{
            "weights_path":   "/code/weights/TC-sign-pytorch-yolov3/best.pt",     ## suppoer model file .pt (pytorch) or .weight (darknet)
            "conf_thresh":    0.5,
            "nms_thresh":     0.5
        },

        "data":{
            "name":           "TC-test",                                          ## Project Name. Output result file will create by using this name. The result file can then be used in the repo `detection_eval` to generate reuslt report.
            "dirpath":        "/data/JPEGImages/test",                            ## datapath
            "isimages":       "True"                                              ## test data is `image` or not. No need to change
        },

        "parameters":{
            "outdir":         "/code/output",                                     ## Output directory.
            "size":           512,                                                ## input size of test. rectangle image will be resize and pad
            "save_images":    "True",                                             ## save result images or not
            "gpus":           ""                                                  ## support multi-gpus
        }

    }
}
```








