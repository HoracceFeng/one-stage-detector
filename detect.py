import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect(config):
    '''
    To start rectangle detect, you should modify manually in `utils/datasets.py` 
        searching `rectangle` and `rectangilar`
    '''

    '''
    (under develop) need to reimplemented and fit `map_eval.py`
        cfg,
        data_cfg,
        weights,
        images='data/samples',  # input folder
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=False
    '''

    ### ARGUMENT
    dictfile             = config["model"]["parameters"]["dictfile"]
    model_format         = config["model"]["weights"]["format"]
    
    weights_path         = config["evaluate"]["model"]["weights_path"]
    conf_thresh          = config["evaluate"]["model"]["conf_thresh"]
    nms_thresh           = config["evaluate"]["model"]["nms_thresh"]

    data_name            = config["evaluate"]["data"]["name"]
    data_path            = config["evaluate"]["data"]["dirpath"]
    isimage              = config["evaluate"]["data"]["isimages"]          ### Not Support Video right now

    outdir               = config["evaluate"]["parameters"]["outdir"]
    img_size             = config["evaluate"]["parameters"]["size"]
    save_images          = config["evaluate"]["parameters"]["save_images"]

    multi_scale          = config["hidden"]["multi_scale"]                 ### Not support yet
    nms_method           = config["hidden"]["nms_method"]
    silent_mode          = config["hidden"]["silent_mode"]


    ### DICTIONARY & PARAMERTERS
    classes = parse_dict_file(dictfile)
    nc = len(classes)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    filename = os.path.join(outdir, data_name+'.txt')
    total_inference_time = 0.

    ### OUTPUT DIR
    device = torch_utils.select_device()
    # if os.path.exists(outdir):
    #     print(">>> Outdir already exists, delete and re-make it ...... ")
    #     shutil.rmtree(outdir)  # delete output folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # make new output folder


    ### INITIALIZE MODEL
    if model_format == 'experiment':
        ### (under develop) need to add func later
        backbone    = config['model']['experiment']['backbone']
        head        = config['model']['experiment']['head']        
        model       = TorchModel()
    elif model_format == 'darknet':
        darknet_cfg = config["model"]["darknet"]["darknet_cfg"]
        head        = config['model']['darknet']['head']        ## support later
        model       = Darknet(darknet_cfg, img_size)
    else:
        print("Model Format Error: Pl ")


    ### WEIGHTS LOAD
    if weights_path.endswith('.pt'):  
        ### pytorch format
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    else:  
        ### darknet format
        _ = load_darknet_weights(model, weights_path)


    ### Fuse and distributed --> Conv2d + BatchNorm2d layers
    model.fuse()
    model.to(device).eval()

    ### DATALOADER --> default square images, will resize to img_size. 
    ###                To support rectangle images, start self.train_rectangular and other `rectangle` flag
    ###                in `utils/datasets.py`. Everything will be fine.
    vid_path, vid_writer = None, None
    if isimage:
        dataloader = LoadImages(data_path, img_size=img_size)
        dataloader.silent = silent_mode
    else:
        ### (deprecated) Right now do not suppoer video format
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)        


    ### EVALUATE SECTION
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        imagename = Path(path).name
        save_path = os.path.join(outdir, 'out_'+imagename)

        ### DETECT
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        ### NMS method can be chosen either 'MERGE' (default), 'AND', 'OR', 'SOFT'
        det = non_max_suppression(pred, conf_thresh, nms_thresh, nms_method)[0]     

        ### RESCALE & INFO
        if det is not None and len(det) > 0:
            # Rescale boxes to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            if not silent_mode:
                print('%gx%g ' % img.shape[2:], end='')  # print image size
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for xmin, ymin, xmax, ymax, conf, cls_conf, cate_id in det:
                boxes = [int(xmin), int(ymin), int(xmax), int(ymax)]   ## the same, I just rewrite again (= =)
                cate = classes[int(cate_id)]
                with open(filename, 'a') as outfile:
                    outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(imagename, 'Out:', cate, conf, boxes))

                # Add bbox to the image
                label = '%s %.2f' % (cate, conf)
                plot_one_box(boxes, im0, label=label, color=colors[int(cate_id)])


        total_inference_time += time.time() - t
        avg_inference_time = total_inference_time / (i+1)
        if not silent_mode:
            print('Done. (%.3fs)  avg_time: (%.3fs)' % (time.time() - t, avg_inference_time))


        ### SAVE RESULT
        if save_images:  
            ### IMAGES
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
            ### (under develop) VIDEO
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    codec = int(vid_cap.get(cv2.CAP_PROP_FOURCC))
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, codec, fps, (width, height))
                vid_writer.write(im0)

    print("==================================================================================")
    print('Results saved to %s' % outdir)
    print('Average Inference Time:', avg_inference_time, 'sec')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration path')
    parser.add_argument('-s', '--silent', type=bool, default=False, help='configuration path')    
    parser_augments = parser.parse_args()
    config_path = parser_augments.config
    silent_mode = parser_augments.silent
    print('>>> Loading config_path ......', config_path)

    ### -------------------------- Main -------------------------- ###
    ### Hyper-Parameters
    config = parse_json_cfg(config_path)
    config['hidden'] = {}

    # (under develop) multi-scale training
    config['hidden']['multi_scale']                = True
    # NMS method can be chosen either 'MERGE' (default), 'AND', 'OR', 'SOFT'
    config['hidden']['nms_method']                 = 'MERGE'
    # silent mode
    config['hidden']['silent_mode']                = silent_mode

    ### Set Environment
    os.environ['CUDA_VISIBLE_DEVICES']  = config['evaluate']['parameters']['gpus']    


    with torch.no_grad():
        detect(config)



