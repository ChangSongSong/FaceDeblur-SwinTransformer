import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.DeblurModel import DeblurModel
from dataset import LoadDataset

from utils import compare_psnr_ssim

class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
        self.n_eps = config['train']['n_epochs']
        self.current_ep = -1
        self.current_loss = 100
        self.save_dir = config['train']['save_dir']

        # Datasets
        train_dataset = LoadDataset(root=config['dataset']['train'], mode='train', size=config['dataset']['size'])
        val_dataset = LoadDataset(root=config['dataset']['val'], mode='val', size=config['dataset']['size'])

        # DataLoader
        self.train_loader = DataLoader(
                                train_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=True
                            )
        self.val_loader = DataLoader(
                                val_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=False
                            )

        # Model
        self.model = DeblurModel(encoder=config['model']['encoder']).to(device=self.device)
        self.optimizer = torch.optim.Adam(
                                self.model.parameters(),
                                lr=config['optimizer']['lr'],
                                weight_decay=config['optimizer']['weight_decay']
                            )

        # loss
        self.criterion = nn.L1Loss().to(device=self.device)

        # tensorboard
        self.writer = SummaryWriter(comment=f'{config["model"]["encoder"]}')

    def train(self):
        print('\n--- start train ---')
        for ep in range(self.current_ep+1, self.n_eps):
            print(f'\nepoch {ep}')
            self.current_ep = ep
            self.train_one_epoch()
            self.validate()

    def train_one_epoch(self):
        ep_loss, ep_psnr, ep_ssim = 0, 0, 0
        loop = tqdm(
                self.train_loader,
                leave=True,
                desc=f"Train Epoch:{self.current_ep}/{self.n_eps}"
            )

        self.model.train()
        for it, (imgs_blur, imgs_GT) in enumerate(loop):
            imgs_blur = imgs_blur.to(device=self.device)
            imgs_GT = imgs_GT.to(device=self.device)

            imgs_deblur = self.model(imgs_blur)
            loss = self.criterion(imgs_deblur, imgs_GT)
            psnr, ssim = compare_psnr_ssim(imgs_deblur, imgs_GT)

            ep_loss += loss.item()/len(self.train_loader)
            ep_psnr += psnr/len(self.train_loader)
            ep_ssim += ssim/len(self.train_loader)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loop.set_postfix(
                    lr=self.optimizer.param_groups[0]['lr'],
                    loss=loss.item(),
                    psnr=psnr,
                    ssim=ssim
                )

        print(f'train      loss: {ep_loss:.3f}, psnr: {ep_psnr:.3f}, ssim: {ep_ssim:.3f}')

        self.writer.add_scalar('Loss/train', ep_loss, self.current_ep)
        self.writer.add_scalar('PSNR/train', ep_psnr, self.current_ep)
        self.writer.add_scalar('SSIM/train', ep_ssim, self.current_ep)

    def validate(self):
        ep_loss, ep_psnr, ep_ssim = 0, 0, 0

        self.model.eval()
        for it, (imgs_blur, imgs_GT) in enumerate(self.val_loader):
            imgs_blur = imgs_blur.to(device=self.device)
            imgs_GT = imgs_GT.to(device=self.device)

            imgs_deblur = self.model(imgs_blur)
            loss = self.criterion(imgs_deblur, imgs_GT)
            psnr, ssim = compare_psnr_ssim(imgs_deblur, imgs_GT)

            ep_loss += loss.item()/len(self.val_loader)
            ep_psnr += psnr/len(self.val_loader)
            ep_ssim += ssim/len(self.val_loader)
        
        print(f'validation loss: {ep_loss:.3f}, psnr: {ep_psnr:.3f}, ssim: {ep_ssim:.3f}')

        self.writer.add_scalar('Loss/val', ep_loss, self.current_ep)
        self.writer.add_scalar('PSNR/val', ep_psnr, self.current_ep)
        self.writer.add_scalar('SSIM/val', ep_ssim, self.current_ep)

        if(ep_loss < self.current_loss):
            self.current_loss = ep_loss
            self.save_checkpoint()

    def save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': self.current_loss,
            'current_ep': self.current_ep,
        }
        checkpoint_path = os.path.join(self.save_dir, 'best.pth')
        torch.save(checkpoint, checkpoint_path)
        print("### Save checkpoint to '{}'".format(checkpoint_path))