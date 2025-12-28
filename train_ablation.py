import torch
import time
import torch.nn
import os
import random
import numpy as np
# from x import Config
import argparse

import logging

# from model.CompFormer import CompFormer
from model.compformer_ablation import CompFormer
from model.criterion_contrastive import Loss as MS3Loss
from model.criterion_s4_contrastive import Loss as S4Loss
from dataloader.ms3_dataset import MS3Dataset
from dataloader.s4_dataset import S4Dataset
from utility import mask_iou, Eval_Fmeasure
from utils.loss_util import LossUtil
from utils import pyutils

from torch.utils.tensorboard import SummaryWriter

import cv2
import torch.nn.functional as F
from einops import rearrange


def getLogger(log_file, name, fmt='%(asctime)s %(levelname)s ==> %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def train(args):
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    os.makedirs(os.path.join(args.log_dir), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, args.dir_name), exist_ok=True)
    log_file = os.path.join(args.log_dir, args.dir_name, f'{log_name}.log')
    logger = getLogger(log_file, __name__)
    # logger.info(f'Load config from {args.cfg}')
    writer = SummaryWriter(args.log_dir)

    if args.session_name == "MS3":
        num_frame = 5
    elif args.session_name == "S4":
        num_frame = 5

    # model 
    vggish=dict(       # 音频编码器模型
        freeze_audio_extractor=True,
        pretrained_vggish_model_path=args.audio_pretrained,
        preprocess_audio_to_log_mel=False,
        postprocess_log_mel_with_pca=False,
        pretrained_pca_params_path=None)
    # mask2former模型
    print(args.if_use_encoder, args.if_use_decoder, args.use_inter_decoder, \
        args.use_intra_decoder, args.use_spatial_encoder, args.if_use_cmfpn, args.use_avfusion, \
            args.use_temporal_mamba, args.use_temporal_encoder
        )
    
    model = CompFormer(d_model=args.d_model, 
                       vggish_config=vggish, 
                       audio_dim=128,
                       num_frame=num_frame,
                       num_classes=1,
                       scale_factor=args.scale_factor,
                       use_vision_backbone=args.use_vision_backbone, 
                       if_use_encoder=args.if_use_encoder, 
                       if_use_decoder=args.if_use_decoder,
                       use_inter_decoder=args.use_inter_decoder, 
                       use_intra_decoder=args.use_intra_decoder, 
                       use_temporal_encoder=args.use_temporal_encoder, 
                       use_spatial_encoder=args.use_spatial_encoder,
                       if_use_cmfpn=args.if_use_cmfpn, 
                       use_temporal_mamba=args.use_temporal_mamba, 
                       use_avfusion=args.use_avfusion, 
                       scan_order=args.scan_order,
                       img_size=args.img_size)
    
    if args.resume:     # 加载之前的训练模型
        model_dict = model.state_dict()
        model_weight = torch.load(args.model_weight, map_location='cpu')
        state_dict = {k: v for k, v in model_dict.items() if k not in model_weight.keys()}
        # print(state_dict)          # 打印出模型中存在但是预训练权重中不存在的结构
        logger.info("Module in the original model: " + str([key for key in state_dict.keys()]))
        
        state_dict = {k: v for k, v in model_weight.items() if k not in model_dict.keys()}
        # print(state_dict)          # 打印出预训练权重中存在但是模型中不存在的结构
        logger.info("Module in the pre-trained model: " + str([key for key in state_dict.keys()]))
        
        if args.use_vision_backbone == 'Vim':
            if args.img_size != 512:      # 由于之前的模型权重是在 512 * 512 的图像上训练的，所以需要对权重进行调整
                for key in model_weight.keys():
                    if key == 'vision_backbone.pos_embed':
                        # 这里对pos_embed进行插值操作
                        pos_tokens = model_weight['vision_backbone.pos_embed']# [:, num_extra_tokens:]
                        embedding_size = pos_tokens.shape[-1]
                        pos_tokens = pos_tokens.view(-1, int(pos_tokens.shape[1] ** 0.5), int(pos_tokens.shape[1] ** 0.5), embedding_size).permute(0, 3, 1, 2)
                        new_pos_embed = F.interpolate(pos_tokens, size=(args.img_size // args.scale_factor, args.img_size // args.scale_factor), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).flatten(1, 2)
                        model_weight['vision_backbone.pos_embed'] = new_pos_embed
    
        model.load_state_dict(model_weight, strict=False)
    model = torch.nn.DataParallel(model).cuda()       # 分布式训练Missing key(s) in state_dict: "visual_fusion_block.multihead_attn.in_proj_weight", "visual_fusion_block.multihead_attn.in_proj_bias", "visual_fusion_block.multihead_attn.out_proj.weight", "visual_fusion_block.multihead_attn.out_proj.bias", "mta_layer.cross_attn_layer.cross_attn.in_proj_weight", "mta_layer.cross_attn_layer.cross_attn.in_proj_bias", "mta_layer.cross_attn_layer.cross_attn.out_proj.weight", "mta_layer.cross_attn_layer.cross_attn.out_proj.bias", "mta_layer.cross_attn_layer.norm1.weight", "mta_layer.cross_attn_layer.norm1.bias", "mta_layer.ffn.linear1.weight", "mta_layer.ffn.linear1.bias", "mta_layer.ffn.linear2.weight", "mta_layer.ffn.linear2.bias", "mta_layer.ffn.norm2.weight", "mta_layer.ffn.norm2.bias", "mta_layer.cross_attn_layers.0.cross_attn.in_proj_weight", "mta_layer.cross_attn_layers.0.cross_attn.in_proj_bias", "mta_layer.cross_attn_layers.0.cross_attn.out_proj.weight", "mta_layer.cross_attn_layers.0.cross_attn.out_proj.bias", "mta_layer.cross_attn_layers.0.norm1.weight", "mta_layer.cross_attn_layers.0.norm1.bias", "mta_layer.ffn_layers.0.linear1.weight", "mta_layer.ffn_layers.0.linear1.bias", "mta_layer.ffn_layers.0.linear2.weight", "mta_layer.ffn_layers.0.linear2.bias", "mta_layer.ffn_layers.0.norm2.weight", "mta_layer.ffn_layers.0.norm2.bias".
    model.train()
    logger.info("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    
    # dataset
    # train_dataset = build_dataset(args.dataset, 'train', args)
    if args.session_name == 'MS3':
        train_dataset = MS3Dataset(split='train', img_size=args.img_size, cfg=args)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_works,
                                                    pin_memory=True)
        max_step = (len(train_dataset) // args.train_batch_size) * args.train_epochs

        # val_dataset = build_dataset(args.dataset, 'test', args)
        val_dataset = MS3Dataset(split='test', img_size=args.img_size, cfg=args)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=args.val_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_works,
                                                    pin_memory=True)
    elif args.session_name == 'S4':
        # train_dataset = build_dataset(args.dataset, 'train', args)
        train_dataset = S4Dataset(split='train', img_size=args.img_size, cfg=args)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_works,
                                                    pin_memory=True)
        max_step = (len(train_dataset) // args.train_batch_size) * args.train_epochs

        # val_dataset = build_dataset(args.dataset, 'test', args)
        val_dataset = S4Dataset(split='test', img_size=args.img_size, cfg=args)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=args.val_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_works,
                                                    pin_memory=True)
    
    # optimizer
    weight_dict = {'dice_loss': 1, 'mix_loss': 0, 'interm_loss': 1, 'focal_loss': 0, 'rec_loss': 0}
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_fscore = pyutils.AverageMeter('F_score')
    loss_util = LossUtil(weight_dict)

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    max_Fscore = 0
    Fscore_list = []
    if not args.eval_only:
        for epoch in range(args.train_epochs):
            if epoch == args.unfreeze_epoch:
                model.module.vision_backbone_frozen(unfrozen=True)
            total_loss = 0
            for n_iter, batch_data in enumerate(train_dataloader):
                if args.session_name == 'MS3':
                    imgs, audio, mask, _ = batch_data
                    mask_num = 5
                    loss = MS3Loss
                elif args.session_name == 'S4':
                    imgs, audio, mask = batch_data
                    mask_num = 1
                    loss = S4Loss

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * mask_num, 1, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                pred_mask, mask_features = model(audio, imgs)       # pred_mask: l, bs * t, 1, h, w     mask_features: bs * t, c, h, w
                loss, loss_dict = loss(pred_mask, mask, weight_dict, mask_features)
                loss_util.add_loss(loss, loss_dict)          # 保存并记录每个batch的loss值
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                global_step += 1
                if (global_step - 1) % 20 == 0:
                    train_log = 'Iter:%5d/%5d, %slr: %.6f' % (
                        global_step - 1, max_step, loss_util.pretty_out(), optimizer.param_groups[0]['lr'])
                    logger.info(train_log)

            writer.add_scalar('total_loss', total_loss / n_iter, epoch)

            # Validation:
            model.eval()
            with torch.no_grad():
                for n_iter, batch_data in enumerate(val_dataloader):
                    # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                    if args.session_name == 'MS3':
                        imgs, audio, mask, _ = batch_data
                    elif args.session_name == 'S4':
                        imgs, audio, mask, _, _ = batch_data
            
                    imgs = imgs.cuda()
                    audio = audio.cuda()
                    mask = mask.cuda()
                    B, frame, C, H, W = imgs.shape
                    imgs = imgs.view(B * frame, C, H, W)
                    mask = mask.view(B * frame, H, W)
                    audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])

                    # [bs*5, 1, 224, 224]
                    output, _ = model(audio, imgs)
                    output = output[-1]          # bs * t, 1, h, w
                    output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
                    miou = mask_iou(output.squeeze(1), mask)             # 计算miou值     bs * t, h, w
                    Fscore = Eval_Fmeasure(output.squeeze(1), mask)
                    avg_meter_miou.add({'miou': miou})
                    avg_meter_fscore.add({'F_score': Fscore})

                miou = (avg_meter_miou.pop('miou'))
                Fscore = (avg_meter_fscore.pop('F_score'))

                writer.add_scalar('miou', miou, epoch)
                writer.add_scalar('F_score', Fscore, epoch)

                if miou > max_miou or Fscore + miou > max_Fscore + max_miou:
                    model_save_path = os.path.join(args.checkpoint_dir, args.dir_name, '%s_best.pth' % (args.session_name))
                    torch.save(model.module.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info('save best model to %s' % model_save_path)

                miou_list.append(miou)
                max_miou = max(miou_list)

                Fscore_list.append(Fscore)
                max_Fscore = max(Fscore_list)

                val_log = 'Epoch: {}, Miou: {}, Fscore: {}, maxMiou: {}, maxFscore: {}'.format(epoch, miou, Fscore, max_miou, max_Fscore)
                logger.info(val_log)

            model.train()
        logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
    
    if args.eval_only:
         # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                
                if args.session_name == 'MS3':
                    imgs, audio, mask, video_name = batch_data
                elif args.session_name == 'S4':
                    imgs, audio, mask, _, video_name = batch_data

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])

                # [bs*5, 1, 224, 224]
                output, _ = model(audio, imgs)
                output = output[-1]          # bs * t, 1, h, w
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
                miou = mask_iou(output.squeeze(1), mask)             # 计算miou值     bs * t, h, w
                Fscore = Eval_Fmeasure(output.squeeze(1), mask)

                # 数据可视化处理
                video_name = video_name[0]
                if args.visualize:
                    pred_masks = output.squeeze(1)
                    pred_masks = rearrange(pred_masks, '(b t) h w -> b t h w', t=frame)      # bs, time, h, w
                    pred_masks = F.interpolate(pred_masks, size=(224, 224), mode='bilinear', align_corners=False)[0]       # time, h, w
                    pred_masks = torch.sigmoid(pred_masks) > 0.5

                    gt_masks = mask.view(B, frame, H, W)     # bs, time, h, w
                    gt_masks = F.interpolate(gt_masks, size=(224, 224), mode='bilinear', align_corners=False)[0]  # time, h, w
                    gt_masks = torch.sigmoid(gt_masks) > 0.5 

                    save_mask_path = os.path.join(args.save_path, video_name + '_' + str(miou.item()))
                    if not os.path.exists(save_mask_path):
                        os.makedirs(save_mask_path)
                    
                    for i in range(len(pred_masks)):       # 对于每一帧图像
                        save_pred_path = os.path.join(save_mask_path, str(i + 1) + '_pred_mask.png')
                        save_gt_path = os.path.join(save_mask_path, str(i + 1) + '_gt_mask.png')
                        img_path = os.path.join(args.video_path, video_name, video_name + '.mp4_' + str(i + 1) + '.png')
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        pred_img_mask = image.copy()
                        pred_mask = pred_masks[i].detach().cpu().numpy()         # h, w
                        pred_img_mask[pred_mask] = (
                                image * 0.5
                                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 246, 143]) * 0.5
                                )[pred_mask]
                        
                        pred_img_mask[pred_mask == False] = (image * 0.5)[pred_mask == False]

                        pred_img_mask = cv2.cvtColor(pred_img_mask, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_pred_path, pred_img_mask)

                        gt_img_mask = image.copy()
                        gt_mask = gt_masks[i].detach().cpu().numpy()      # h, w
                        gt_img_mask[gt_mask] = (
                                image * 0.5
                                + gt_mask[:, :, None].astype(np.uint8) * np.array([255, 246, 143]) * 0.5
                                )[gt_mask]
                        gt_img_mask[gt_mask == False] = (image * 0.5)[gt_mask == False]

                        gt_img_mask = cv2.cvtColor(gt_img_mask, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_gt_path, gt_img_mask)

                avg_meter_miou.add({'miou': miou})
                avg_meter_fscore.add({'F_score': Fscore})
                logger.info('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, Fscore))

            miou = (avg_meter_miou.pop('miou'))
            Fscore = (avg_meter_fscore.pop('F_score'))

            logger.info(f'test miou: {miou.item()}')
            logger.info(f'test F_score: {Fscore}')
            logger.info('test miou: {}, F_score: {}'.format(miou.item(), Fscore))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='work_dirs_vmamba/ms3', help='log dir')
    parser.add_argument('--dir_name', type=str, default='avs_ms3', help='log dir name')
    parser.add_argument('--checkpoint_dir', type=str, default='work_dirs_vmamba/ms3', help='dir to save checkpoints')
    parser.add_argument("--session_name", default='MS3', type=str, help="the MS3 setting")
    parser.add_argument('--vision_pretrained', help='dir to vision checkpoints', 
                        default='pretrained_backbones/pvt_v2_b5.pth', type=str)
    parser.add_argument('--img_size', default=512, type=int)
                        
    parser.add_argument('--use_vision_backbone', default='Vim', type=str)
    parser.add_argument('--lr', help='learning rate', default=2e-5, type=float)        # 原来是2e-5, no weight decay
    parser.add_argument('--audio_pretrained', help='dir to audio checkpoints', 
                        default='/bin/pretrained_backbones/vggish-10086976.pth', type=str)
    parser.add_argument('--dataset', default='MS3Dataset', type=str, help='the dataset to train')        # 目前先支持MS3和S4
    parser.add_argument('--data_dir', default='Multi-sources', type=str)
    parser.add_argument('--visualize', default=False, type=bool)

    parser.add_argument('--train_epochs', default=120, type=int)
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--val_batch_size', default=1, type=int)
    parser.add_argument('--num_works', default=8, type=int)
    parser.add_argument('--loss_type', default='dice', type=str)
    parser.add_argument('--unfreeze_epoch', default=20, type=int)
    parser.add_argument('--eval_only', default=False, type=bool)

    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--scale_factor', default=4, type=int)
    parser.add_argument('--if_use_encoder', action='store_true')
    parser.add_argument('--if_use_decoder', action='store_true')
    parser.add_argument('--use_inter_decoder', action='store_true')
    parser.add_argument('--use_intra_decoder', action='store_true')
    parser.add_argument('--use_temporal_encoder', action='store_true')
    parser.add_argument('--use_spatial_encoder', action='store_true')
    parser.add_argument('--if_use_cmfpn', action='store_true')
    parser.add_argument('--use_temporal_mamba', action='store_true')
    parser.add_argument('--use_avfusion', action='store_true')
    parser.add_argument('--scan_order', default=8, type=int)

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_weight', default='', type=str)

    args = parser.parse_args()
    train(args)
