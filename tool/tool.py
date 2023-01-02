from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import os
import random
import numpy as np
import shutil
import torch
import collections
from torch.autograd import Variable
import math
import cv2

# 用来给图片画框图
def plot_boxes(img, boxes):
    '''plot boxes on image
    '''
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4],
                     box[5], box[6], box[7]], outline=(0, 255, 0))
    return img


# 将图片resize后标准化
def resizeNormalize(size, img):
    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ColorJitter(
                                        0.5, 0.5, 0.5, 0.25),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return transform(img)


def normalize(img):
    r'''只normalize不resize'''
    t = transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return t(img)


def split_train_and_test_set(dataset_name, dataset_path, radio=9):
    r'''
    dataset_name表示分割的数据集名称
    dataset_path表示跟数据集有关的总目录
    radio是分割比率，正整数和浮点数都行，10最大，0最小，radio为9时代表训练集占据数据集的九成
    '''

    input_path = os.path.join(dataset_path, r'temp', dataset_name)
    input_paths = []
    input_paths.append(os.path.join(input_path, r'img'))
    input_paths.append(os.path.join(input_path, r'gt'))
    output_paths = []
    output_paths.append(os.path.join(dataset_path, r'train_img'))
    output_paths.append(os.path.join(dataset_path, r'test_img'))
    output_paths.append(os.path.join(dataset_path, r'train_gt'))
    output_paths.append(os.path.join(dataset_path, r'test_gt'))

    input_img_list = np.array(os.listdir(input_paths[0]))
    input_gt_list = np.array(os.listdir(input_paths[1]))
    # 开始分割数据集
    index_list = list(range(len(input_img_list)))
    random.shuffle(index_list)
    train_index = int(len(input_img_list)*radio/10)
    input_file_list = []
    input_file_list.append(input_img_list[index_list[:train_index]])
    input_file_list.append(input_img_list[index_list[train_index:]])
    input_file_list.append(input_gt_list[index_list[:train_index]])
    input_file_list.append(input_gt_list[index_list[train_index:]])

    for i in range(len(output_paths)):
        # 创建输出文件夹
        if os.path.exists(output_paths[i]):
            shutil.rmtree(output_paths[i])
        os.makedirs(output_paths[i])
        for j in range(len(input_file_list[i])):
            # 获得源文件路径和目的地路径
            out_file = os.path.join(output_paths[i], input_file_list[i][j])
            in_file = os.path.join(input_paths[i//2], input_file_list[i][j])
            shutil.copy2(in_file, out_file)
    print('The training set and test set of \033[1;33m{}\033[0m are constructed'.format(
        dataset_name))


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'# for `-1~index

        self.dict = {}
        for i,char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank ' required by wrap_ctc
            self.dict[char] = i +1

    def encode(self, text):
        """将文字转为对应的下标，改了
        Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        # length = [len(s) for s in text]
        # text=[ self.alphabet_dict[i]  for s in text for i in s]
        # return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """将下标转化为文字
        Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class Rotate(object):

    def __init__(self, coordinate, image=None):
        self.coordinate = coordinate
        self.xy = [tuple(self.coordinate[k]) for k in ['left_top', 'right_top', 'right_bottom', 'left_bottom']]
        self._mask = None
        if image is not None:
            self.image = image.convert('RGB')
            self.image.putalpha(self.mask)

    @property
    def mask(self):
        if not self._mask:
            mask = Image.new('L', self.image.size, 0)
            draw = ImageDraw.Draw(mask, 'L')
            draw.polygon(self.xy, fill=255)
            self._mask = mask
        return self._mask

    def run(self):
        image = self.rotation_angle()
        box = image.getbbox()
        return image.crop(box)

    def rotation_angle(self):
        x1, y1 = self.xy[0]
        x2, y2 = self.xy[1]
        angle = self.angle([x1, y1, x2, y2], [0, 0, 10, 0]) * -1
        return self.image.rotate(angle, expand=True)

    def get_angle(self):
        x1, y1 = self.xy[0]
        x2, y2 = self.xy[1]
        angle = self.angle([x1, y1, x2, y2], [0, 0, 10, 0]) * -1
        return angle

    def angle(self, v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle


def crop_img(index,img,result_file):
    r'''裁剪图片
        index: 8*1的ndarray, 坐标
        img: PIL.Image 图片
        result_file: 裁剪图片后存储文件路径名称
        不返回值'''
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    var1=np.reshape(index,(4,2))
    var1 = {'left_bottom':var1[0],'left_top': var1[1],
        'right_top':var1[2],'right_bottom': var1[3] }
    rotate = Rotate(var1,img)
    crop_img=rotate.run().convert('RGB')
    crop_img.save(result_file)
    return


def text_process(CRNN_ans):
    r'''文本处理
        CRNN_ans: dict, key是jiazhao, hesuan, xingchengka.
            val是list, 为文字信息s
        返回一个字典. 格式为: { 'color':[1: 绿色, 2: 黄色, 3: 红色]
        '途径城市':'', '核酸检测时间':'',  '核酸检测机构':''}'''
    def format_date(text):
        ans=''
        for val in text:
            try :
                int(val)
            except :
                continue
            ans+=val
        ans='{}-{}-{}  {}:{}:{}'.format(ans[0:4],ans[4:6],ans[6:8],ans[8:10],ans[10:12],ans[12:14])
        return ans

    # 肯定得有个文字转换map
    OCR_map={}
    ans={}
    for i,inf in enumerate(CRNN_ans['xingchengka']):
        if '省' in inf:
            ans['途径城市']=inf+CRNN_ans['xingchengka'][i+1]
        if '色' in inf and '行程卡' in inf:
            if '绿' in inf:
                ans['color']=1
            elif '黄' in inf:
                ans['color']=2
            elif '红' in inf:
                ans['color']=3
    for i,inf in enumerate(CRNN_ans['hesuan']):
        if '检测时间' in inf:
            ans['核酸检测时间']=format_date(CRNN_ans['hesuan'][i+1])
        if '检测机构' in inf:
            ans['核酸检测机构']=CRNN_ans['hesuan'][i+1]
        

    return CRNN_ans


def boxes_process(boxes,img_size,y_thresh=20):
    r'''坐标处理
        boxes: 二维ndarray
        y_thresh: 当y轴不相差y_thresh时认为是同一行
        返回已经排序好的坐标, list, 三维, 第一维代表每一行, 第二维代表每一行中的框元素, 
            为ndarray, 第三维是坐标'''
    def fit_long(boxes,img_size,gate_length=225,gate_radio=0.8):
        r'''EAST有些长文本覆盖不全, 加长度阈值判断然后手动扩长
            boxes: 已经排序好的坐标, list, 三维, 第一维代表每一行, 第二维代表每一行中的框元素, 
                第三维是坐标
            gate_length: 长文本长度阈值
            gate_radio: 缩放比率, 取值0~1, 为0.8时代表再往外扩张0.2倍
            返回已经扩张且排序好的坐标, list, 三维, 第一维代表每一行, 第二维代表每一行中的框元素, 
                第三维是坐标'''
        for items in boxes:
            for item in items:
                length=(item[6]-item[0]+item[4]-item[2])/2
                # 大于阈值，增长长度
                if length>gate_length:
                    x=(item[4]-item[2])*((1-gate_radio)/gate_radio)
                    item[4]= min(item[4]+x,img_size[0])
                    item[2]= max(item[2]-x,0)
                    x=(item[6]-item[0])*((1-gate_radio)/gate_radio)
                    item[6]=min(item[6]+x,img_size[0])
                    item[0]=max(item[0]-x,0)

        return boxes
    ans=[]
    # 先把一整行的放一块
    for box in boxes:
        box=get_coordinate(box)
        var1=np.reshape(box,(4,2))
        var1 = {'left_bottom':var1[0],'left_top': var1[1],
            'right_top':var1[2],'right_bottom': var1[3] }
        rotate = Rotate(var1)
        angle=rotate.get_angle()
        if abs(angle)>15:
            continue
        flag=True
        mean_box=(box[1]+box[3]+box[5]+box[7])/4
        for i,items in enumerate(ans):
            item=items[0]
            mean_item=(item[1]+item[3]+item[5]+item[7])/4
            if abs(mean_item-mean_box)<y_thresh:
                ans[i].append(box)
                flag=False
                break
        if flag:
            ans.append([])
            ans[len(ans)-1].append(box)

    # 按行排序
    ans.sort(key=lambda item: (item[0][1]+item[0][3]+item[0][5]+item[0][7])/4)
    # 每行中再按照列排序
    for box in ans:
        box.sort(key=lambda item: item[0])

    ans=fit_long(ans,img_size)
    return ans


def get_coordinate(coordinate):
    var1=[0]*4
    coordinate=np.array(coordinate).reshape(4,2)
    x_min_index=np.argmin(coordinate[:,0])
    x_min=coordinate[x_min_index][0]
    max_=int(1e10)
    for i in range(len(coordinate)):
        if abs(coordinate[i][0]-x_min)<max_ and i !=x_min_index:
            max_=coordinate[i][0]-x_min
            x_index=i
    if coordinate[x_min_index][1]>coordinate[x_index][1]:
        var1[3]=coordinate[x_min_index]
        var1[0]=coordinate[x_index]
    else:
        var1[0]=coordinate[x_min_index]
        var1[3]=coordinate[x_index]
    a_set=set([0,1,2,3])
    a_set.discard(x_min_index)
    a_set.discard(x_index)
    a_list=list(a_set)
    if coordinate[a_list[0]][1]>coordinate[a_list[1]][1]:
        var1[2]=coordinate[a_list[0]]
        var1[1]=coordinate[a_list[1]]
    else:
        var1[1]=coordinate[a_list[0]]
        var1[2]=coordinate[a_list[1]]

    var1=np.reshape(var1,(-1))
    var1=np.array([var1[6],var1[7],var1[0],var1[1],var1[2],var1[3],
        var1[4],var1[5]])
    # var1 = {'left_top': var1[0], 'right_top':var1[1],'right_bottom': var1[2], 'left_bottom':var1[3]}
    return var1


def sample_pics_by_video(video_file:str,save_path:str,frame_step=2):
    r'''从视频中采样图片
        video_file: 视频文件路径
        save_path: 存储路径
        frame_step: 采样间隔, 每隔多少个帧采样, 默认每隔2个帧
        不返回东西'''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(video_file)
    fps=cap.get(cv2.CAP_PROP_FPS)
    flag, frame = cap.read()
    cnt=0
    while flag:
        if cnt%(frame_step+1)==0: 
            cv2.imwrite(os.path.join(save_path,'{}.png'.format(cnt)), frame)  # 存储为图像 
        cnt += 1
        flag, frame = cap.read()
    cap.release()
    return