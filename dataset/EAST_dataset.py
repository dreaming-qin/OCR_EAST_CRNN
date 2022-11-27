import random
from cv2 import resize
from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import shutil
from config import EAST_config as EAST_cfg


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            index1  : offset of point1
            index2  : offset of point2
            r       : [r1, r2, r3, r4] in paper
            coef    : shrink ratio in paper
    Output:
            vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(
        vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            coef    : shrink ratio in paper
    Output:
            v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
            cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:	
            vertices: vertices of text region <numpy.ndarray, (8,)>
            theta   : angle in radian measure
            anchor  : fixed position during rotation
    Output:
            rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
            the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
            err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
        cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
            the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
            (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(
        list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, size):
    '''get rotated locations of all pixels for next stages
    Input:
            rotate_mat: rotatation matrix
            anchor_x  : fixed x position
            anchor_y  : fixed y position
            length    : length of image
    Output:
            rotated_x : rotated x positions <numpy.ndarray, (length,length)>
            rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(size[0])
    y = np.arange(size[1])
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
        np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def get_score_geo(img, vertices, labels, scale):
    '''generate score gt and geometry gt
    Input:
            img     : PIL Image
            vertices: vertices of text regions <numpy.ndarray, (n,8)>
            labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
            scale   : feature map / image
            length  : image length
    Output:
            score gt, geo gt, ignored
    '''
    score_map = np.zeros(
        (int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros(
        (int(img.height * scale), int(img.width * scale), 5), np.float32)
    ignored_map = np.zeros(
        (int(img.height * scale), int(img.width * scale), 1), np.float32)

    index_x = np.arange(0, img.width, int(1/scale))
    index_y = np.arange(0, img.height, int(1/scale))
    index_x, index_y = np.meshgrid(index_x, index_y)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_polys.append(
                np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue

        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))
                         ).astype(np.int32)  # scaled & shrinked
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(
            rotate_mat, vertice[0], vertice[1], img.size)

        d1 = scale*(rotated_y - y_min)
        d1[d1 < 0] = 0
        d2 = scale*(y_max - rotated_y)
        d2[d2 < 0] = 0
        d3 = scale*(rotated_x - x_min)
        d3[d3 < 0] = 0
        d4 = scale*(x_max - rotated_x)
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask

    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)
    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1), torch.Tensor(ignored_map).permute(2, 0, 1)


def extract_vertices(lines):
    '''extract vertices info from txt lines
    Input:
            lines   : list of string info
    Output:
            vertices: vertices of text regions <numpy.ndarray, (n,8)>
            labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
    '''
    labels = []
    vertices = []
    for line in lines:
        a = line.rstrip('\n').lstrip('\ufeff').split(',')[:8]
        a = list(map(float, a))
        a = list(map(int, a))
        vertices.append(a)
        # label = 0 if '###' in line else 1
        label = 1
        labels.append(label)
    return np.array(vertices), np.array(labels)


def create_equal_ratio_points(points, ratio, gravity_point):
    """
    @brief      创建等比例的点
    @param      points         为list,The points
    @param      ratio          The ratio
    @param      gravity_point  The gravity point
    @return     { description_of_the_return_value }
    """
    # 判断输入条件
    if len(points) <= 2 or not gravity_point:
        return list()

    new_points = list()
    length = len(points)

    for i in range(length):
        vector_x = points[i][0] - gravity_point[0]
        vector_y = points[i][1] - gravity_point[1]
        new_point_x = ratio * vector_x + gravity_point[0]
        new_point_y = ratio * vector_y + gravity_point[1]
        new_point = [new_point_x, new_point_y]
        new_points.append(new_point)
    return new_points


def img_adjust(img, vertices, size=(512, 512)):
    # from detect import resize_img
    # img, ratio_y, ratio_x = resize_img(img)

    ratio_y = size[1] / img.size[1]
    ratio_x = size[0] / img.size[0]
    for vertice in vertices:
        for i in range(0, 8, 2):
            vertice[i] *= ratio_x
            vertice[i+1] *= ratio_y

    return img.resize(size), vertices


class custom_dataset(data.Dataset):
    def __init__(self, img_path, gt_path, scale=0.25, length=512):
        super(custom_dataset, self).__init__()
        self.img_files = [os.path.join(img_path, img_file)
                          for img_file in sorted(os.listdir(img_path))]
        self.gt_files = [os.path.join(gt_path, gt_file)
                         for gt_file in sorted(os.listdir(gt_path))]
        self.scale = scale
        self.length = length

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        with open(self.gt_files[index], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        vertices, labels = extract_vertices(lines)

        img = Image.open(self.img_files[index]).convert('RGB')
        img, vertices = img_adjust(img, vertices, (EAST_cfg.imgW, EAST_cfg.imgH))
        # img, vertices = adjust_height(img, vertices)
        # img, vertices = rotate_img(img, vertices)
        # img, vertices = crop_img(img, vertices, labels, self.length)

        # from tool.common import plot_boxes
        # plot_img=plot_boxes(img,vertices)
        # plot_img.save('./result/IMG_2074.JPG')

        transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        # 里面已经对gt进行了缩放操作，方法名是shrink_poly
        score_map, geo_map, ignored_map = get_score_geo(
            img, vertices, labels, self.scale)

        # unloader = transforms.ToPILImage()
        # iii = unloader(score_map)
        # iii.save('./result/IMG_2074.JPG')

        return transform(img), score_map, geo_map, ignored_map


if __name__ == '__main__':
    img_path = '../dataset/train_img/T1_kR_XadkXXcDMjo8_100900.jpg.jpg'
    gt_path = '../dataset/train_gt/T1_kR_XadkXXcDMjo8_100900.jpg.txt'

    with open(gt_path, encoding='utf-8') as f:
        boxes = f.readlines()
    boxes = np.array([box.split(',') for box in boxes])
    boxes = boxes[:, :-1]
    boxes = [list(map(float, box)) for box in boxes]
    from PIL import Image
    img = Image.open(img_path).convert('RGB')

    from tool.common import plot_boxes
    # 画图
    plot_img = plot_boxes(img, boxes)

    # 缩放图
    var1 = []
    for box in boxes:
        box = np.reshape(box, (-1, 2))
        gravity_point = np.mean(box, axis=0).tolist()
        box = create_equal_ratio_points(box, 0.7, gravity_point)
        var1.append(box)
    boxes = np.reshape(var1, (-1, 8))
    plot_img = plot_boxes(plot_img, boxes)

    plot_img = plot_img.resize((512, 512))
    plot_img.save('./result/IMG_2074.JPG')
