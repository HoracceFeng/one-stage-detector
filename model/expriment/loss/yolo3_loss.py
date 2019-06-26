# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.box_utils import match, log_sum_exp


class YoloLoss(nn.Module):
    """SSD Weighted Loss Function

    priorboxes = [ -1, [cx, cy, w, h] ]

    Compute Targets:
        loss_xy = 

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, priors, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, obj_data = predictions
        num = loc_data.size(0)
        priors = self.priors
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))   ## num_priors == predict_box_num
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        pos = conf_t > 0
        # num_pos = pos.sum()

        ## Localization Loss (MSE, loss_xy, loss_wh)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)   

        loc_p_xy = loc_p[:, 0:2]
        loc_t_xy = lox_t[:, 0:2]
        loss_l_xy = F.mse_loss(torch.sigmoid(loc_p_xy, loc_t_xy), size_average=False)

        loc_p_wh = loc_p[:, 2:4]
        loc_t_wh = loc_t[:, 2:4]
        loss_l_xy = F.mse_loss(loc_p_wh, loc_t_wh, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = F.binary_cross_entropy(batch_conf, conf_t.view(-1,1), size_average=False)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # objectness loss
        batch_objn = obj_data.view(-1, 1)
        loss_o = F.binary_cross_entropy_with_logits(batch_objn, torch.zeros_like(batch_objn.shape), size_average=False)

        return loss_l, loss_c, loss_o
