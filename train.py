import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from tool.common import plot_boxes
from tool.common import split_train_and_test_set
import os
import time
import numpy as np
from PIL import Image
from config import EAST_config as EAST_cfg, CRNN_config as CRNN_cfg



def train_EAST(dataset_path, dataset_name, pths_path,
          batch_size, lr, num_workers, epoch_iter,
          model_path=None,is_generate_dataset=True):
    from model import EAST
    from eval import eval_EAST
    from dataset.EAST_dataset import custom_dataset
    from loss import EASTLoss

    if is_generate_dataset:
        split_train_and_test_set(dataset_name, dataset_path,radio=9.5)
    train_img_path = os.path.join(dataset_path, r'train_img')
    train_gt_path = os.path.join(dataset_path, r'train_gt')

    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True)

    criterion = EASTLoss(weight_angle=4,weight_classify=7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': lr}],lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.8)

    min_loss = int(2e4)
    for epoch in range(epoch_iter):
        model.train()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(
                device), gt_geo.to(device), ignored_map.to(device)
            pred_score, pred_geo = model(img)
            
            loss = criterion(gt_score, pred_score, gt_geo,
                             pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))

        scheduler.step()
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(
            epoch_loss, time.time()-epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('='*50)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(
                pths_path, 'east_epoch{}_loss{}.pth'.format(epoch+1, int(min_loss))))

            # test
            test_img_path = os.path.join(EAST_cfg.dataset_path,'EAST','demo')
            result_path = EAST_cfg.result_path
            ans=eval_EAST(model, test_img_path, result_path)
            img,boxes=ans[0]
            if boxes is not None:
                # 画图
                plot_img = plot_boxes(img, boxes)
                result_file = os.path.join(result_path, 'epoch{}.jpg'.format(epoch))
                plot_img.save(result_file)
    


if __name__ == '__main__':
    dataset_path='./data/EAST'
    dataset_name = r'天池ICPR'
    is_generate_dataset=False
    pths_path = EAST_cfg.pth_path
    batch_size = EAST_cfg.batch
    lr = EAST_cfg.lr
    num_workers = EAST_cfg.workers
    epoch_iter = EAST_cfg.epoch
    model_path = os.path.join(EAST_cfg.pth_path,'east_epoch21_loss634.pth') 
    model_path=None
    train_EAST(dataset_path, dataset_name, pths_path,
          batch_size, EAST_cfg.lr, num_workers, epoch_iter,
          model_path,is_generate_dataset)
