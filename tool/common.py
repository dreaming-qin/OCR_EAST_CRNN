from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import os
import random
import numpy as np
import shutil
import torch
import collections
from torch.autograd import Variable

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
