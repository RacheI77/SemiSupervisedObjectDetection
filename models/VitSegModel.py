# @Time : 2023/6/20 18:00
# @Author : Li Jiaqi
# @Description :
import os
from models.ViT_Decoder import Decoder
from models.ViT_EncoderDecoder import EncoderDecoder
import models.Loss as myLoss
import torch.nn as nn
import torch
import config


class VitSegModel(nn.Module):
    def __init__(self, pretrain_weight='vit-seg-without-autoencoder epoch 38 train 0.230 eval 0.274 fps 0.70.pth',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.encoder_model.cuda()
        self.decoder = Decoder(img_size=(config.ModelConfig['imgh'], config.ModelConfig['imgw']), out_chans=1,
                               patch_size=self.encoder_model.patch_size, depth=self.encoder_model.n_blocks,
                               embed_dim=self.encoder_model.embed_dim, num_heads=self.encoder_model.num_heads).cuda()
        self.model = EncoderDecoder(self.encoder_model, self.decoder)

        # load the pretrained weights
        self.model.load_state_dict(torch.load(os.path.join('checkpoints', pretrain_weight)))
        print('pretrained model loaded')

        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, self.model.parameters()),
                                          lr=config.ModelConfig['lr'], weight_decay=config.ModelConfig['weight_decay'],
                                          betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.ModelConfig['scheduler'])
        # loss_function = torch.nn.L1Loss()
        self.loss_function = myLoss.SegmentationLoss(1, loss_type='dice', activation='none')
        self.activation_fn = torch.nn.Sigmoid()

    def predict(self, batch_images):
        mask = None
        mask = self.activation_fn(mask)
        pass

    def train_one_epoch(self, imgs, masks):
        loss=0
        self.train_from_loss(loss)
        pass

    def train_from_loss(self, loss):
        pass

    def scheduler_step(self):
        self.scheduler.step()

    def show_mask(self, img, mask):
        pass
