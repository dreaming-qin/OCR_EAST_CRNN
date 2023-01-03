import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
import os
import time
import numpy as np
from PIL import Image
import random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


from config import EAST_config as EAST_cfg, CRNN_config as CRNN_cfg
from tool.tool import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    
def train_CRNN(dataset_path , pths_path,
          batch_size, lr, num_workers, epoch_iter,
          model_path=None):
    from model import CRNN
    from dataset.CRNN_dataset import customDataset,alignCollate

    # batch train方法
    def trainBatch(net, criterion, optimizer):
        data = next(train_iter)
        image, text = data

        batch_size = image.size(0)
        text, length = converter.encode(text)
        text, length, image = text.to(device), length.to(device), image.to(device)

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size)).to(device)
        
        torch.backends.cudnn.enabled = False
        cost = criterion(preds, text, preds_size, length)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        return cost
    

    # val方法
    def val(net, criterion, max_iter=10):
        test_dataset = customDataset(os.path.join(dataset_path,'test'),'gt.list')
        print('Start val')

        net.eval()
        data_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=True, batch_size=batch_size, num_workers=int(num_workers),
            collate_fn=alignCollate(imgH=CRNN_cfg.imgH, imgW=CRNN_cfg.imgW, keep_ratio=CRNN_cfg.keep_ratio))
        val_iter = iter(data_loader)

        i = 0
        n_correct = 0
        loss_avg = averager()

        max_iter = min(max_iter, len(data_loader))
        cnt=1
        for i in range(max_iter):
            data = next(val_iter)
            i += 1
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            image = cpu_images.to(device)
            t, l = converter.encode(cpu_texts)
            text = t.to(device)
            length = l.to(device)

            preds = crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text, preds_size, length)
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            cpu_texts = converter.decode(text.data, length, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                a_set=set(target.lower())
                for s in pred:
                    if s.lower() in a_set:
                        n_correct+=1
                cnt+=len(target)

        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[
            :n_test_disp]
        f=open(os.path.join(CRNN_cfg.result_path,'val.txt'),'a',encoding='utf8')
        str1='lr:{}\n'.format( optimizer.param_groups[0]['lr'])
        str1+='epoch {}\n'.format(epoch+1)
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            str1+='%-20s => %-20s, gt: %-20s \n' % (raw_pred, pred, gt)

        accuracy = n_correct / cnt
        str1+='Test loss: %f, accuray: %f \n' % (loss_avg.val(), accuracy)
        str1+='--------------------------------------------------------------------------------------------------\n\n'
        print(str1)
        f.writelines(str1)
        f.close()

        return accuracy

    # 在结果.txt中打印多少个样例
    n_test_disp=10
    # 随机种子
    manualSeed=1234
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # 中文字典
    alphabet = ''
    with open(os.path.join(dataset_path,'char_std.txt'), encoding='utf8') as f:
        alphabet = f.readlines()
    alphabet = [line.strip('\n') for line in alphabet]
    alphabet = ''.join(alphabet)
    # 中文字典个数
    nclass = len(alphabet) + 1
    # 预训练路径
    pretrained=model_path
    # 创建路径
    if not os.path.exists(pths_path):
        os.makedirs(pths_path)
    if not os.path.exists(CRNN_cfg.result_path):
        os.makedirs(CRNN_cfg.result_path)
    cudnn.benchmark = True
    # crnn输入通道数
    input_channel = 3

    # 训练用
    converter = strLabelConverter(alphabet,ignore_case=True)

    if torch.cuda.is_available():
        print("note: you have gpu")

    train_dataset = customDataset(os.path.join(dataset_path,'train'),'gt.list')
    assert train_dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=alignCollate(imgH=CRNN_cfg.imgH, imgW=CRNN_cfg.imgW, keep_ratio=CRNN_cfg.keep_ratio))

    criterion = nn.CTCLoss().to(device)


    crnn = CRNN.CRNN(CRNN_cfg.imgH, input_channel, nclass, CRNN_cfg.hidden_size)
    crnn = crnn.to(device)
    if pretrained != None:
        print('loading pretrained model from %s' % pretrained)
        crnn.load_state_dict(torch.load(pretrained))
    print(crnn)

    # setup optimizer
    optimizer = torch.optim.Adam(crnn.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

    image = torch.FloatTensor(batch_size, input_channel, CRNN_cfg.imgW, CRNN_cfg.imgH)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)


    
    crnn.train()
    max_acc = -1

    for epoch in range(epoch_iter):
        train_iter = iter(train_loader)
        i = 0
        epoch_loss = 0
        epoch_time = time.time()
        min_iter=min(int(6e3),len(train_loader))
        while i < min_iter:
            batch_time=time.time()
            cost = trainBatch(crnn, criterion, optimizer)
            epoch_loss+=cost.item()

            print('[%d/%d][%d/%d] Loss: %f time:%.9f' %
                    (epoch+1, epoch_iter, i+1, min_iter, cost.item(),time.time()-batch_time))
            i += 1
        
        # val
        accuracy=val(crnn, criterion)
        # do checkpointing
        if max_acc < accuracy:
            max_acc = accuracy
            torch.save(
                crnn.state_dict(), os.path.join(pths_path, 
                'CRNN_epoch{}_loss{}_acc{}.pth'.format(epoch+1, int(epoch_loss),max_acc)))
            

        if epoch>15 and optimizer.param_groups[0]['lr']>1e-5:
            scheduler.step()
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(
                epoch_loss, time.time()-epoch_time))


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
