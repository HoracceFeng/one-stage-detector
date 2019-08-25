import argparse
import json

from torch.utils.data import DataLoader

# from models import *
from model.darknet.models import *
from utils.datasets import *
from utils.utils import *


def test(config, model=None):
    '''
    degrade as test module and only use during model training
    for evaluation and service, please use `detect.py`
    '''

    ### ARGUMENT
    dictfile             = config["model"]["parameters"]["dictfile"]
    debug                = config["model"]["parameters"]["debug"]            

    model_format         = config["model"]["weights"]["format"]
    pretrain_model       = config["model"]["weights"]["pretrain"]
    resume               = config["model"]["weights"]["resume"]
    freeze_backbone      = config["model"]["weights"]["freeze"]
    transfer             = config["model"]["weights"]["transfer"]

    data_format          = config["test"]["data"]["format"]
    data_filepath        = config["test"]["data"]["txtpath"]
    batch_size           = config["test"]["parameters"]["batch"]
    img_size             = config["test"]["parameters"]["size"]


    ### PARAMETERS & DICTIONARY
    save_json  = False
    iou_thres  = 0.5
    nms_thres  = 0.5
    conf_thres = 0.01
    classes    = parse_dict_file(dictfile)
    nc         = len(classes)
    names      = classes
    colors     = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    test_path  = data_filepath

    ### MODEL INIT
    assert model is not None
    device = next(model.parameters()).device  # get model device


    ### Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=2,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)


    ### Test Section
    seen = 0
    model.eval()
    print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        ### Plot images with bounding boxes
        if debug and batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, fname='test_batch0.jpg')

        ### Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        ### Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss_i, _ = compute_loss(train_out, targets, model)
            loss += loss_i.item()

        ### Run NMS  --> (under develop) will add more NMS methods
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres, nms_style='MERGE')

        ### Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': classes[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float(d[4])
                    })

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    ### Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        try:
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        except:
            print('RuntimeWarning: Mean of empty slice --> invalid value encountered in double_scalars')
            mp, mr, map, mf1 = 0., 0., 0., 0.

    ### Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    ### Save JSON for COCO only
    # if save_json and map and len(jdict):
    #     imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
    #     with open('results.json', 'w') as file:
    #         json.dump(jdict, file)

    #     from pycocotools.coco import COCO
    #     from pycocotools.cocoeval import COCOeval

    #     # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #     cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
    #     cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

    #     cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    #     cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()
    #     map = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # Return results
    return mp, mr, map, mf1, loss / len(dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('-c', '--config', type=str, help='configure path')
    opt = parser.parse_args()
    config_path = opt.config
    print(opt)

    config = parse_json_cfg(config_path)

    with torch.no_grad():
        mAP = test(config)
        print(mAP)
