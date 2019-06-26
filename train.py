import argparse, time, torch, json
import numpy as np
from datetime import datetime

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
# from models import *
from model.darknet.models import *
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import *



def train(config, evolve=None):

    ### ARGUMENT
    dictfile             = config["model"]["parameters"]["dictfile"]
    autoevolve           = config["model"]["parameters"]["autoevolve"]      ## Not support yet 
    notest               = config["model"]["parameters"]["notest"]
    nosave               = config["model"]["parameters"]["nosave"]
    save_interval        = config["model"]["parameters"]["save_interval"]
    debug                = config["model"]["parameters"]["debug"]           ## under develop 

    model_format         = config["model"]["weights"]["format"]
    pretrain_model       = config["model"]["weights"]["pretrain"]
    resume               = config["model"]["weights"]["resume"]
    freeze_backbone      = config["model"]["weights"]["freeze"]
    transfer             = config["model"]["weights"]["transfer"]

    name                 = config["train"]["name"]

    data_format          = config["train"]["data"]["format"]
    data_filepath        = config["train"]["data"]["txtpath"]
    augment              = config["train"]["data"]["augment"]

    epochs               = config["train"]["parameters"]["max_epoch"]
    batch_size           = config["train"]["parameters"]["batch"]
    img_size             = config["train"]["parameters"]["size"]

    learning_rate        = config["train"]["parameters"]["learning_rate"]
    lr_scheduler         = config["train"]["parameters"]["lr_scheduler"]    ## Not support yet
    momentum             = config["train"]["parameters"]["momentum"]
    weight_decay         = config["train"]["parameters"]["weight_decay"]

    min_lr               = config["hidden"]["min_lr"]
    accumulate           = config["hidden"]["accumulate"]
    multi_scale          = config["hidden"]["multi-scale"]
    num_workers          = config["hidden"]["num_workers"]
    mixed_precision      = config["hidden"]["mixed_precision"]

    ### PARAMETERS
    start_epoch          = 0
    best_loss            = float('inf')
    DISTRIBUTED          = False

    if transfer == "" or not str(transfer).isdigit():
        cutoff           = -1  # backbone reaches to cutoff layer
    else:
        cutoff           = int(transfer)


    ### DICTIONARY
    classes = parse_dict_file(dictfile)
    nc = len(classes)

    ### DEVICE
    device = torch_utils.select_device()
    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
        num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale


    ### OUTPUT PATH
    init_seeds()
    weights_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights', name)
    latest_weights = os.path.join(weights_dir, 'latest.pt')
    best_weights   = os.path.join(weights_dir, 'best.pt')

    # weight directory
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(">>> weights_dir: create successfully ...... ", weights_dir)
    else:
        print(">>> weights_dir: already exists ...... ", weights_dir)

    # log directory
    logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print(">>> logdir: create successfully ...... ", logdir)
    else:
        print(">>> logdir: already exists ...... ", logdir)


    ### #*# NETWORK DEFINCE 
    if model_format == "experiment":
        ### (under develop) need to add func later
        backbone    = config['model']['experiment']['backbone']
        head        = config['model']['experiment']['head']        
        model       = TorchModel()

    elif model_format == "darknet":
        darknet_cfg = config["model"]["darknet"]["darknet_cfg"]
        head        = config['model']['darknet']['head']        ## support later
        model       = Darknet(darknet_cfg, img_size).to(device)
    else:
        print("[X] Model_Format_Error: ['model']['weights']['format'] should be `darknet` or `experiment` ")
    
    if head == 'yolov3':
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)


    ### #*# LOSS WEIGHT LOAD
    if evolve is None:
        # (optional)(under develop) loss weight parameters path, change it by using `evolve.py`
        loss_weight_param_path = os.path.join(weights_dir, 'loss_weights_parameters.json')
        if not os.path.exists(loss_weight_param_path):
            lwfile = open(loss_weight_param_path, 'w')
            lwhyp = {
                     "head":           head,
                     "k":              1,  
                     "xy":             0.25,  
                     "wh":             0.25,  
                     "cls":            0.25,  
                     "conf":           0.25,  
                     "iou_thresh":     0.5,        
                    }
            json.dump(lwhyp, lwfile)
            print(">>> loss_weight_parameters_file: created automatically ......", loss_weight_param_path)
        else:
            lwhyp = parse_json_cfg(loss_weight_param_path)
            print(">>> loss_weight_parameters_file: already exists ......", loss_weight_param_path)
    else:
        lwhyp  = evolve


    ### #*# NETWORK WEIGHTS LOAD
    print('\n')
    if resume:
        assert os.path.exists(pretrain_model) == True
        print(">>> Load Pretrain Weight:", pretrain_model)
        # resume by using pretrain model, the ['model']['format'] should match
        # load pytorch weights
        if pretrain_model.split('.')[-1] in ['pt', 'pth']:
            ckpt = torch.load(pretrain_model, map_location=device)
            # partial model, 
            # now only for backbone loading
            # patial layer cut by `cutoff` will be added later
            if cutoff > -1:
                model.load_state_dict({k: v for k, v in ckpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
                for p in model.parameters():
                    if head == 'yolov3':
                        p.requires_grad = True if p.shape[0] == nf else False   
            # whole model              
            else:
                model.load_state_dict(ckpt['model'])
        # load darknet weights
        else:
            cutoff = load_darknet_weights(model, pretrain_model, cutoff=cutoff)    ## cutoff = -1 use whole layers
    else:
        print(">>> No Pretrain Model. Using initial weights")
        if False:
            ### #*# initial model using Kaiming OR Xavier, skip to use Random Uniform 
            weights_init(model)


    ### #*# NETWORK OPTIMIZATION
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    ### #*# SCHEDULER -- self define lf and scheduler --> https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** ( np.log10(min_lr/learning_rate) * (1 - x / epochs) * x / epochs)  # exp ramp
    lf = lambda x: 1 - 10 ** ( np.log10(min_lr/learning_rate) * (1 - x / epochs) )  # inverse exp ramp
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch-1)


    ### #*# DATASET DEFINE
    train_path = data_filepath
    if data_format in ["darknet", "voc"]:
        dataset = LoadImagesAndLabels(train_path, img_size, batch_size, augment=augment, _format=data_format, _dict=classes)
    else:
        print('DataFormatError: You should either choose `darknet` or `voc` in ["train"]["data"]["format"] ')


    ### DATALOADER
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,  # disable rectangular training if True
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)


    ### DISTRIBUTED TRAINING DEFINE -- deprecated. To use it, please set the related parameters here
    if DISTRIBUTED and torch.cuda.device_count() > 1:
        _backend    = 'nccl'
        _dist_url   = 'tcp://127.0.0.1:9999'
        _world_size = 1
        _rank       = 0
        dist.init_process_group(backend=_backend, init_method=_dist_url, world_size=_world_size, rank=_rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)


    ### MIXED PRECISION TRAINING -- deprecated. https://github.com/NVIDIA/apex
    # install help: https://github.com/NVIDIA/apex/issues/259
    if mixed_precision:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')


    ### TRAIN SECTION
    t, t0 = time.time(), time.time()
    if head == lwhyp['head']:
        model.lwhyp = lwhyp  # attach hyperparameters to model
    else:
        print('LossWeight Error: The loss weight parameters cannot be loaded since the detector head not match. \
            The detector head is [', head, ']Please check the ["loss_weights"]["head"] in ', loss_weight_param_path)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model)
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches, warmup
    if debug:
        os.remove(os.path.join(logdir, 'check_train_image.jpg')) if os.path.exists(os.path.join(logdir, 'check_train_image.jpg')) else None
        os.remove(os.path.join(logdir, 'check_test_image.jpg')) if os.path.exists(os.path.join(logdir, 'check_test_image.jpg')) else None

    for epoch in range(start_epoch, epochs):
        ### Start traing, pass the train flag to model
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
        ### Update scheduler
        scheduler.step()
        ### Freeze backbone 
        if freeze_backbone and cutoff > 0:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  
                    p.requires_grad = False if epoch == 0 else True

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            if i % 10 != 0:
                continue
            imgs = imgs.to(device)
            targets = targets.to(device)
            nt = len(targets)

            ### Plot images with bounding boxes
            if debug and epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, fname=os.path.join(logdir, 'check_train_batch-0.jpg'))

            ### SGD burn-in / warmup
            if epoch == 0 and i <= n_burnin:
                lr = learning_rate * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            ### Run model
            pred = model(imgs)

            ### Compute loss
            if head == 'yolov3':
                loss, loss_items = compute_loss(pred, targets, model)   ## YOLO_loss, deprecated later
            else:
                print('DetectorError: please provide the correct name of detector head in ', model_format)

            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            ### Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            ### Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            ### Update running mean of tracked metrics
            mloss = (mloss * i + loss_items) / (i + 1)

            ### Print batch results
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss, nt, time.time() - t)
            t = time.time()
            print(s)

            ### Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 21)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        ### (deprecated) Calculate mAP (always test final epoch, skip first 5 if nosave)
        if not evolve:
            if not (notest or (nosave and epoch < 5)) or epoch == epochs - 1:
                with torch.no_grad():
                    results = test.test(config, model)

        ### Write epoch results
        _date_ = datetime.now()
        month  = '%.2d' % _date_.month
        day    = '%.2d' % _date_.day
        hour   = '%.2d' % _date_.hour
        minute = '%.2d' % _date_.minute
        logpath = os.path.join(logdir, 'results_'+month+day+'_'+hour+minute+'.txt')
        with open(logpath, 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        ### Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        ### Save training results
        save = (not nosave) or (epoch == epochs - 1)
        if save:
            ### Create checkpoint
            ckpt = {'epoch': epoch,
                    'best_loss': best_loss,
                    'model': model.module.state_dict() if type( \
                        model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                    'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(ckpt, latest_weights)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(ckpt, best_weights)

            # Save backup every `save_interval` epochs
            if epoch > 0 and epoch % save_interval == 0:
                weights_name = os.path.join(weights_dir, name+'_Epoch_'+str(epoch)+'.pt')
                torch.save(ckpt, weights_name)

            # Delete checkpoint
            del ckpt

    dt = (time.time() - t0) / 3600
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch, dt))
    return results, lwhyp


def print_mutation(lwhyp, results):
    hyp = lwhyp.copy()
    if 'head' in hyp.keys():
        hyp.pop('head')
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,                 help='configure path')
    parser.add_argument('-e', '--evolve', type=bool, default=False, help='evolve loss weight paramters only')
    parser_augments = parser.parse_args()
    config_path = parser_augments.config
    evolve = parser_augments.evolve
    print('>>> Loading config_path ......', config_path)

    ### -------------------------- Main -------------------------- ###
    ### Hyper-Parameters
    config = parse_json_cfg(config_path)
    config['hidden'] = {}
    # minimum learning rate as default
    config['hidden']['min_lr']                     = 1e-6  ## lr * (10 ** lrf)
    # accumerate batch for later gradient updates
    config['hidden']['accumulate']                 = 1
    # Dataloader num_workers
    config['hidden']['num_workers']                = 2
    # multi-scale training
    config['hidden']['multi-scale']                = True
    # Mixed precision training: https://github.com/NVIDIA/apex --> install https://github.com/NVIDIA/apex/issues/259
    config['hidden']['mixed_precision']            = False
    if evolve:    
        config["train"]["parameters"]["max_epoch"] = 1
    # show config
    # print(config)

    ### Set Environment
    os.environ['CUDA_VISIBLE_DEVICES']  = config['train']['parameters']['gpus']

    ### Train
    results, lwhyp = train(config)

    ### #*# Evolve (can adjust evolve standard)
    if evolve:
        print("Starting Evolve ......")
        print("Loss Weight Parameters", lwhyp)
        # (evolve standard) use mAP for fitness
        # can be adjusted by your necessity
        best_fitness = results[2]    #*# results: P, R, mAP, F1, test_loss

        ### Write mutation results
        print_mutation(lwhyp, results)

        gen = 5  # generations to evolve
        for _ in range(gen):

            ### #*# Mutate hyperparameters (mutate not more than 20%) --> can adjust hyper-param search space
            old_lwhyp = lwhyp.copy()
            init_seeds(seed=int(time.time()))

            keys = list(lwhyp.keys())
            keys.remove('head')            
            for i, k in enumerate(keys):
                if str(lwhyp[k]).isdigit:
                    x = (np.random.randn(1) * 0.2 + 1) ** 1.1  # plt.hist(x.ravel(), 100)
                    lwhyp[k] = lwhyp[k] * float(x)  # vary by about 30% 1sigma

            ### (not recommend) adjust `iou_thresh`, `momentum`, `weight_decay`
            # keys = ['iou_thresh'] # 'momentum', 'weight_decay'
            # limits = [(0, 0.90), (0.80, 0.95), (0, 0.01)]
            # for k, v in zip(keys, limits):
            #     hyp[k] = np.clip(hyp[k], v[0], v[1])

            ### Normalize loss components (sum to 1)
            # keys = ['xy', 'wh', 'cls', 'conf']
            keys = list(lwhyp.keys())
            for rmkey in ['k', 'head', 'iou_thresh']:
                keys.remove(rmkey)
            s = sum([v for k, v in lwhyp.items() if k in keys])
            for k in keys:
                lwhyp[k] /= s

            ### Determine mutation fitness
            results, _ = train(config, evolve=lwhyp)
            mutation_fitness = results[2]

            ### Write mutation results
            print_mutation(lwhyp, results)

            # Update hyperparameters if fitness improved
            if mutation_fitness > best_fitness:
                # Fitness improved!
                print('Fitness improved!')
                best_fitness = mutation_fitness
            else:
                lwhyp = old_lwhyp.copy()  # reset hyp to



