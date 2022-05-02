import argparse
import yaml
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Input a config file.')
parser.add_argument('--config', help='Config file path')
args = parser.parse_args()
f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

import numpy as np
from torch.utils.data import DataLoader
import cv2
from dataset import TrainDataset
import models
import torch
import loss

if config['use_wandb']:
    import wandb
    wandb.init(project=config['project'], entity=config['entity'], name=config['run_name'])


os.makedirs(config['checkpoint_folder'], exist_ok=True)
model = torch.nn.DataParallel(models.mana(config,is_training=True)).cuda()


if config['hot_start']:
    checkpt=torch.load(config['hot_start_checkpt'])
    model.module.load_state_dict(checkpt)

loss_fn = loss.lossfun()

train_dataset = TrainDataset(config['dataset_path'], patch_size=config['patch_size'], scale=4)
dataloader = DataLoader(dataset=train_dataset,
                        batch_size=config['batchsize'],
                        shuffle=True,
                        num_workers=config['num_workers'],
                        pin_memory=True)


count = 0
stage1=config['stage1']
stage2=config['stage2']
for epoch in range(0, config['epoch']):
    with tqdm(dataloader, desc="Training MANA") as tepoch:
        for inp, gt in tepoch:
            tepoch.set_description(f"Training MANA--Epoch {epoch}")
            if count==0:
                for p in model.module.parameters():
                    p.requires_grad=True
                for p in model.module.nonloc_spatial.W_z1.parameters():
                    p.requires_grad=False
                model.module.nonloc_spatial.mb.requires_grad=False
                model.module.nonloc_spatial.D.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=1e-4, betas=(0.5, 0.999))
            elif count==stage1:
                for p in model.module.parameters():
                    p.requires_grad=False
                for p in model.module.nonloc_spatial.W_z1.parameters():
                    p.requires_grad=True
                model.module.nonloc_spatial.mb.requires_grad=True
                model.module.nonloc_spatial.D.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=1e-4, betas=(0.5, 0.999))
            elif count==stage2:
                for p in model.module.parameters():
                    p.requires_grad=True
                model.module.nonloc_spatial.mb.requires_grad=True
                model.module.nonloc_spatial.D.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=1e-4, betas=(0.5, 0.999))
    
    
            inp = inp.float().cuda()
            gt = gt.float().cuda()
            
            optimizer.zero_grad()
            oup,qloss = model(inp)
            
            if count<stage1:
                loss = loss_fn(gt, oup)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage1'})
                if config['use_wandb']:
                    wandb.log({"L1 Loss": loss})
                
                    
            elif count<stage2:
                loss=torch.mean(qloss)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix({'Quantize Loss:': loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage2'})
                if config['use_wandb']:
                    wandb.log({"Quantize Loss": loss})
                
            else:
                loss= loss_fn(gt, oup)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage3'})
                if config['use_wandb']:
                    wandb.log({"L1 Loss": loss})
    
            count += 1
    
            if count % config['N_save_checkpt'] == 0:
                tepoch.set_description("Training MANA--Saving Checkpoint")
                torch.save(model.module.state_dict(), config['checkpoint_folder'] + config['checkpoint_name'])
                if config['save_samples']:
                    cv2.imwrite(config['checkpoint_folder'] + 'input.png', np.flip(inp[0, inp.shape[1]//2, :, :, :].permute(1, 2, 0).cpu().numpy(),2)*255)
                    cv2.imwrite(config['checkpoint_folder'] + 'super_res.png', np.flip(oup[0, :, :, :].permute(1, 2, 0).data.cpu().numpy(),2) * 255)
                    cv2.imwrite(config['checkpoint_folder'] + 'ground_truth.png', np.flip(gt[0, :, :, :].permute(1, 2, 0).cpu().numpy(),2) * 255)
            
