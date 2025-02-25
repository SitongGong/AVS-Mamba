import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)  # [bs*5, 1, 224, 224]

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    '''
    inputs: bs * t, 1, h, w
    targets: bs * t, 1, h, w
    '''
    # prob = inputs.sigmoid()
    indices = torch.tensor(list(range(0, len(inputs), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(inputs, dim=0, index=indices)  # [bs, 1, 224, 224]
    targets = targets.flatten(0, 1).unsqueeze(1)
    
    ce_loss = F.binary_cross_entropy_with_logits(first_pred, targets, reduction="none")
    
    prob = first_pred.sigmoid()

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).mean()

def F5_Dice_loss(pred_mask, first_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    indices = torch.tensor(list(range(0, len(pred_mask), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(pred_mask, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

    pred_mask = first_pred.flatten(2).flatten(0, 1)   # bs x T, h x w
    gt_mask = first_gt_mask.flatten(2).flatten(0, 1)   # bs x T, h x w
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()

def contrastive_loss(origin_audio_feat, audio_feat, tau):
    # origin_audio_feat: bs, t, dim
    # audio_feat: bs, t, dim

    origin_audio_feat = origin_audio_feat.flatten(0, 1)            # bs * t, dim
    audio_feat = audio_feat.flatten(0, 1)              # bs * t, dim

    origin_audio_feat = F.normalize(origin_audio_feat, dim=-1)       # 归一化
    audio_feat = F.normalize(audio_feat, dim=-1)

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))
    logit_scale = torch.clamp(logit_scale.exp(), max=100)          # 可学习参数调节

    BT = audio_feat.shape[0]
    logits = torch.arange(BT).long().to(audio_feat.device)

    loss_origin_audio = origin_audio_feat @ audio_feat.t()         # bs * t, bs * t
    loss_audio_feat = audio_feat @ origin_audio_feat.t()           # bs * t, bs * t
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    contrastive_loss = cross_entropy_loss(loss_origin_audio * logit_scale, logits) + cross_entropy_loss(loss_audio_feat * logit_scale, logits)

    return contrastive_loss


def contrastive_loss_modi(origin_audio_feat, audio_feat, tau):
    # origin_audio_feat: bs, t, dim
    # audio_feat: bs, t, dim
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))
    logit_scale = torch.clamp(logit_scale.exp(), max=100)          # 可学习参数调节

    logits = torch.arange(1).long().to(audio_feat.device)

    bs, t = audio_feat.shape[0], audio_feat.shape[1]
    origin_audio_feat = origin_audio_feat.flatten(0, 1)            # bs * t, dim
    audio_feat = audio_feat.flatten(0, 1)              # bs * t, dim

    origin_audio_feat = F.normalize(origin_audio_feat, dim=-1)
    audio_feat = F.normalize(audio_feat, dim=-1)

    loss_origin_audio = origin_audio_feat @ audio_feat.t()         # bs * t, bs * t
    loss_audio_feat = audio_feat @ origin_audio_feat.t()           # bs * t, bs * t
    # loss_origin_audio = origin_audio_feat.reshape(bs, t, -1)          # bs, time, bs * time
    
    cross_entropy_loss = nn.CrossEntropyLoss()

    contrastive_loss1 = 0

    for idx in range(loss_audio_feat.shape[0]):
        if idx // t == 0:
            cat_audio_feat = torch.cat((loss_audio_feat[idx][idx].unsqueeze(0), loss_audio_feat[idx][(idx // t + 1) * t:]))
        elif idx // t == bs - 1:
            cat_audio_feat = torch.cat((loss_audio_feat[idx][idx].unsqueeze(0), loss_audio_feat[idx][: (idx // t) * t]))
        else:
            cat_audio_feat = torch.cat((loss_audio_feat[idx][idx].unsqueeze(0), loss_audio_feat[idx][: (idx // t) * t], loss_audio_feat[idx][(idx // t + 1) * t:]))

        contrastive_loss1 = contrastive_loss1 + cross_entropy_loss(cat_audio_feat.unsqueeze(0) * logit_scale, logits)

    contrastive_loss2 = 0

    for idx in range(loss_origin_audio.shape[0]):
        if idx // t == 0:
            cat_audio_feat = torch.cat((loss_origin_audio[idx][idx].unsqueeze(0), loss_origin_audio[idx][(idx // t + 1) * t:]))
        elif idx // t == bs - 1:
            cat_audio_feat = torch.cat((loss_origin_audio[idx][idx].unsqueeze(0), loss_origin_audio[idx][: (idx // t) * t]))
        else:
            cat_audio_feat = torch.cat((loss_origin_audio[idx][idx].unsqueeze(0), loss_origin_audio[idx][: (idx // t) * t], loss_origin_audio[idx][(idx // t + 1) * t:]))
        
        contrastive_loss2 = contrastive_loss2 + cross_entropy_loss(cat_audio_feat.unsqueeze(0) * logit_scale, logits)

    contrastive_loss = contrastive_loss1 / (bs * t) + contrastive_loss2 / (bs * t)

    return contrastive_loss


def Loss(pred_mask, gt_mask, weight_dict, mask_features):
    # pred_mask: l, bs * t, dim, h, w          # 改为直接用dice_loss和focal_loss的
    # mask_features: bs * t, dim, h * w
    # gt_mask: bs x t, 1, h, w
   
    # rec_loss = nn.MSELoss(reduction='mean')(mask_feature, audio_feat)

    mask_feature = torch.mean(mask_features, dim=1, keepdim=True)         # bs * t, 1, h * w
    # size = mask_feature.shape[-1]
    # mask_feature = mask_feature.reshape(-1, 1, int(math.sqrt(size)), int(math.sqrt(size)))      # bs * t, 1, h, w   
    mask_feature = F.interpolate(mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)

    interm_masks = pred_mask[: -1]
    pred_mask = pred_mask[-1]

    total_loss = 0
    dice_loss = F5_Dice_loss(pred_mask, gt_mask)
    mask_loss = F1_IoU_BCELoss(pred_mask, gt_mask)
    mix_loss = weight_dict['dice_loss'] * F5_Dice_loss(mask_feature, gt_mask) +\
         weight_dict['focal_loss'] * sigmoid_focal_loss(mask_feature, gt_mask)
    
    interm_loss = 0
    for interm_mask in interm_masks:
        interm_loss += F5_Dice_loss(interm_mask, gt_mask) * weight_dict['dice_loss']
        interm_loss += F1_IoU_BCELoss(interm_mask, gt_mask) * weight_dict['focal_loss']

    total_loss = interm_loss * weight_dict['interm_loss'] + mix_loss * weight_dict['mix_loss'] + \
        weight_dict['dice_loss'] * dice_loss + weight_dict['focal_loss'] * mask_loss

    loss_dict = {'focal_loss': mask_loss.item(), 'dice_loss': dice_loss.item(), 'mix_loss': mix_loss.item(), 
                 'interm_loss': interm_loss.item(), 'total_loss': total_loss.item()}

    return total_loss, loss_dict