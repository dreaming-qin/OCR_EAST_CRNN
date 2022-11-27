import time
import torch
import subprocess
import os
from model.EAST import EAST
import numpy as np
import shutil
from PIL import Image, ImageDraw
from tool.common import plot_boxes, normalize
from config import EAST_config as EAST_cfg, CRNN_config as CRNN_cfg
import lanms
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_EAST(model, test_img_path, result_path):
    r'''返回list, 元素是(图片，点坐标)
    model: nn.Module
    test_img_path: test的文件目录
    result_path: 存储结果路径
    '''
    def detect_dataset(model, test_img_path, result_path):
        '''detection on whole dataset, save .txt results in submit_path
        Input:
                model        : detection model
                test_img_path: dataset path
                submit_path  : submit result for evaluation
        '''
        img_files = os.listdir(test_img_path)
        img_files = [os.path.join(test_img_path, img_file)
                     for img_file in img_files]

        for i, img_file in enumerate(img_files):
            print('evaluating {} image'.format(i), end='\r')
            img = Image.open(img_file).convert('RGB')
            boxes = detect(img, model, device)
            ans.append((img, boxes))
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
        img, ratio_h, ratio_w = resize_img(img,size=(EAST_cfg.imgW,EAST_cfg.imgH))
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

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    model.eval()
    start_time = time.time()
    detect_dataset(model, test_img_path, result_path)
    print('eval time is {}'.format(time.time()-start_time))

    return ans


if __name__ == '__main__':
    
    ccc = os.listdir('.\pth\EAST')
    for sss in ccc:
        if 'east_epoch24_loss594' not in sss:
            continue
        model_path = os.path.join(EAST_cfg.pth_path, sss)
        model = EAST(False).to(device)
        model.load_state_dict(torch.load(
            model_path, map_location=device))
        test_img_path = os.path.join(EAST_cfg.dataset_path, 'demo')
        result_path = EAST_cfg.result_path
        ans = eval_EAST(model, test_img_path, result_path)
        img, boxes = ans[0]
        if boxes is not None:
            # 画图
            plot_img = plot_boxes(img, boxes)
            result_file = os.path.join(
                result_path, sss.replace('.pth', '.jpg'))
            plot_img.save(result_file)
