import math
import os

import time
import lanms
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.autograd import Variable
import cv2
import torch.nn as nn

from config import VGGface_config as face_cfg
from config import CRNN_config as CRNN_cfg
from config import EAST_config as EAST_cfg
from model.CRNN import CRNN
from model.EAST import EAST
from tool.tool import *



def eval_EAST(model: EAST, test_img, device, result_path=None):
    r'''返回list, 元素是(图片，点坐标)
    model: nn.Module
    test_img: 测试图片文件
    result_path: 存储结果路径, 画好图的, 为None代表不存结果
    '''
    def detect_dataset(model, test_img, result_path):
        '''detection on whole dataset, save .txt results in submit_path
        Input:
                model        : detection model
                test_img: test img path
                submit_path  : submit result for evaluation
        '''
        img_files = [test_img]

        for i, img_file in enumerate(img_files):
            print('evaluating {} image'.format(i), end='\r')
            img = Image.open(img_file).convert('RGB')
            boxes = detect(img, model, device)
            var_img = Image.open(img_file).convert('RGB')
            ans.append((var_img, boxes))
            if result_path is not None:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                if boxes is not None:
                    # 画图
                    plot_img = plot_boxes(img, boxes)
                    result_file = os.path.join(
                        result_path, os.path.basename(img_file))
                    plot_img.save(result_file)
            # with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
            # 	f.writelines(seq)

    def detect(img, model, device):
        '''detect text regions of img using model
        Input:
                img   : PIL Image
                model : detection model
                device: gpu if gpu is available
        Output:
                detected polys
        '''
        img, ratio_h, ratio_w = resize_img(
            img, size=(EAST_cfg.imgW, EAST_cfg.imgH))
        with torch.no_grad():
            score, geo = model(normalize(img).unsqueeze(0).to(device))

        boxes = get_boxes(score.squeeze(0).cpu().numpy(),
                          geo.squeeze(0).cpu().numpy())
        return adjust_ratio(boxes, ratio_w, ratio_h)

    def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
        '''get boxes from feature map
        Input:
                score       : score map from model <numpy.ndarray, (1,row,col)>
                geo         : geo map from model <numpy.ndarray, (5,row,col)>
                score_thresh: threshold to segment score map
                nms_thresh  : threshold in nms
        Output:
                boxes       : final polys <numpy.ndarray, (n,9)>
        '''
        score = score[0, :, :]
        xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
        if xy_text.size == 0:
            return None

        xy_text = xy_text[np.lexsort((xy_text[:, 1], xy_text[:, 0]))]
        valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
        valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
        polys_restored, index = restore_polys(
            valid_pos, valid_geo, score.shape)
        if polys_restored.size == 0:
            return None

        boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = polys_restored
        boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
        return boxes

    def adjust_ratio(boxes, ratio_w, ratio_h):
        '''refine boxes
        Input:
                boxes  : detected polys <numpy.ndarray, (n,9)>
                ratio_w: ratio of width
                ratio_h: ratio of height
        Output:
                refined boxes
        '''

        if boxes is None or boxes.size == 0:
            return None
        boxes[:, [0, 2, 4, 6]] /= ratio_w
        boxes[:, [1, 3, 5, 7]] /= ratio_h
        return np.around(boxes)

    def resize_img(img, size=(512, 512)):
        r'''resize image to be divisible by 32
        如果用size=(512,512)的话会有很严重的误判
        '''
        # w, h = img.size
        # img=img.resize(size)
        # ratio_h=size[1]/h
        # ratio_w=size[0]/w
        w, h = img.size
        resize_w = w
        resize_h = h

        resize_h = int(resize_h / 32) * 32
        resize_w = int(resize_w / 32) * 32
        img = img.resize((resize_w, resize_h), Image.BILINEAR)
        ratio_h = resize_h / h
        ratio_w = resize_w / w

        return img, ratio_h, ratio_w

    def get_rotate_mat(theta):
        '''positive theta value means rotate clockwise'''
        return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    def is_valid_poly(res, score_shape, scale):
        '''check if the poly in image scope
        Input:
                res        : restored poly in original image
                score_shape: score map shape
                scale      : feature map -> image
        Output:
                True if valid
        '''
        cnt = 0
        for i in range(res.shape[1]):
            if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                    res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
                cnt += 1
        return True if cnt <= 1 else False

    def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
        '''restore polys from feature maps in given positions
        Input:
                valid_pos  : potential text positions <numpy.ndarray, (n,2)>
                valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
                score_shape: shape of score map
                scale      : image / feature map
        Output:
                restored polys <numpy.ndarray, (n,8)>, index
        '''
        polys = []
        index = []
        valid_pos *= scale
        d = valid_geo[:4, :]*scale  # 4 x N
        angle = valid_geo[4, :]  # N,

        for i in range(valid_pos.shape[0]):
            x = valid_pos[i, 0]
            y = valid_pos[i, 1]
            y_min = y - d[0, i]
            y_max = y + d[1, i]
            x_min = x - d[2, i]
            x_max = x + d[3, i]
            rotate_mat = get_rotate_mat(-angle[i])

            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0, :] += x
            res[1, :] += y

            if is_valid_poly(res, score_shape, scale):
                index.append(i)
                polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1],
                              res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
        return np.array(polys), index

    ans = []

    model.eval()
    start_time = time.time()
    detect_dataset(model, test_img, result_path)
    print('eval time is {}'.format(time.time()-start_time))

    return ans


def eval_CRNN(model: CRNN, test_img, converter, device, result_path=None):
    r'''返回解码字符串str
    model: nn.Module
    test_img: 测试图片路径
    convert: 字符串编解码转换器
    result_path: 存储结果路径, 显示图片对应的文字存到test.txt文件中, 为None代表不存储
    '''
    img_path = test_img

    image = Image.open(img_path).convert('RGB')
    image = resizeNormalize((CRNN_cfg.imgH, CRNN_cfg.imgW), image)

    # unloader = transforms.ToPILImage()
    # iii = unloader(image)
    # iii.save('.aaa.jpg')

    image = image.to(device)
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    if result_path is not None:
        with open(os.path.join(result_path, 'test.txt'), 'a', encoding='utf-8') as f:
            f.writelines('%-20s => %-20s \n' % (raw_pred, sim_pred))
    # print('%-20s => %-20s ' % (raw_pred, sim_pred))
    return sim_pred


def eval_VGGface(model: nn.Sequential, img_file_one, img_file_two,device, result_path=None):
    r''''
        model: VGGface模型
        img_one: 第一张人脸图片路径
        img_two: 第二张人脸图片路径
        result_path: 结果输出路径, 为None时代表不输出
        返回置信度, 为浮点数
    '''
    def img_editor(img):
        img = cv2.resize(img, face_cfg.img_size)
        imgp = np.zeros((3, face_cfg.img_size[0], face_cfg.img_size[1]))
        temp = [img[:, :, i]-img[:, :, i].mean() for i in range(3)]
        for i in range(3):
            imgp[i, :, :] = temp[i]
        imgp = torch.FloatTensor(imgp)
        return imgp
    
    
    def euc(a, b):
        # 计算欧式距离
        return np.linalg.norm(a-b)


    img_one = cv2.imread(img_file_one)
    img_one = img_editor(img_one)
    img_two = cv2.imread(img_file_two)
    img_two = img_editor(img_two)
    with torch.no_grad():
        vec_one = model(img_one.unsqueeze(0).to(device)).view(-1).detach().numpy()
        vec_two = model(img_two.unsqueeze(0).to(device)).view(-1).detach().numpy()
    dis = euc(vec_one, vec_two)
    if result_path is not None:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file=os.path.join(result_path,'eval.txt')
        str1='img1: {} \n img2: {} \n score:{} \n\n\n'.format(img_file_one,img_file_two,dis)
        with open(result_file,'a',encoding='utf-8') as f:
            f.writelines(str1)
    return dis


if __name__ == '__main__':
    EAST_flag = False
    CRNN_flag = False
    VGGface_flg = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # EAST------------------------------------------------------------
    if EAST_flag:
        ccc = os.listdir('.\pth\EAST')
        for sss in ccc:
            # if 'east_epoch24_loss594' not in sss:
            #     continue
            model_path = os.path.join(
                EAST_cfg.pth_path, 'east_epoch18_loss687.pth')
            model = EAST(False).to(device)
            model.load_state_dict(torch.load(
                model_path, map_location=device))
            test_img = os.path.join(
                EAST_cfg.dataset_path, 'demo/xingchengka.jpg')
            result_path = EAST_cfg.result_path

            # ans = eval_EAST(model, test_img, result_path)
            # img, boxes = ans[0]
            # for ij,box in enumerate(boxes):
            #     # 画图
            #     plot_img = plot_boxes(img, [box])
            #     result_file = os.path.join(result_path, '{}.png'.format(ij))
            #     plot_img.save(result_file)

            img = Image.open(test_img).convert('RGB')
            boxes = np.loadtxt('bbb.txt')
            for ij, box in enumerate(boxes):
                if ij == 31:
                    aaa = 1
                box = get_coordinate(box)
                # 画图
                result_file = os.path.join(
                    result_path, 'temp/{}.png'.format(ij))
                crop_img(np.array(box), img, result_file)
                # 画图
                plot_img = plot_boxes(img, [box])
                result_file = os.path.join(result_path, '{}.png'.format(ij))
                plot_img.save(result_file)
    # EAST结束-------------------------------------------------------------

    # CRNN-----------------------------------------------------------------
    if CRNN_flag:
        alphabet = ''
        with open(os.path.join(CRNN_cfg.dataset_path, 'char_std.txt'), encoding='utf8') as f:
            alphabet = f.readlines()
        alphabet = [line.strip('\n') for line in alphabet]
        alphabet = ''.join(alphabet)
        converter = strLabelConverter(alphabet)

        model = CRNN.CRNN(CRNN_cfg.imgH, 3, len(
            alphabet) + 1, CRNN_cfg.hidden_size)
        model_path = os.path.join(
            CRNN_cfg.pth_path, 'CRNN_epoch1005_loss1073_acc0.8327645051194539.pth')
        model = model.to(device)
        print('loading pretrained model from %s' % model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))

        result_path = CRNN_cfg.result_path
        test_img = os.path.join(CRNN_cfg.dataset_path, 'demo/demo.png')

        eval_CRNN(model, test_img, result_path, converter)
    # CRNN结束-------------------------------------------------------------
