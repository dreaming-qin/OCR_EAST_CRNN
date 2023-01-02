import torch
import os
import pickle


from model import CRNN
from model.EAST import EAST
from config import CRNN_config as CRNN_cfg
from config import EAST_config as EAST_cfg
from tool.common import *
from eval import eval_CRNN,eval_EAST



#整合EAST和CRNN，完成文字识别
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def OCR(jiazhao_img_file,hesuan_img_file,xingchengka_img_file):
    r'''识别图片文字
        jiazhao_img_file: 驾照图片路径
        hesuan_img_file: 核酸图片路径
        xingchengka_img_file: 行程卡图片路径
        输出: 返回一个字典. 格式为: {'name':'', 
            'color':[1: 绿, 2: 黄色, 3: 红色]}'''
    
    img_files={'xingchengka':xingchengka_img_file,
        'hesuan':hesuan_img_file,'jiazhao':jiazhao_img_file}

    # 加载EAST模型------------------------------------------------------------
    EAST_model = EAST(False).to(device)
    EAST_model.load_state_dict(torch.load(
        os.path.join(EAST_cfg.pth_path,'east_epoch18_loss687.pth'), map_location=device))
    # EAST结束---------------------------------------------------------------

    # 加载CRNN模型-----------------------------------------------------------------
    alphabet = ''
    with open(os.path.join(CRNN_cfg.dataset_path, 'char_std.txt'), encoding='utf8') as f:
        alphabet = f.readlines()
    alphabet = [line.strip('\n') for line in alphabet]
    alphabet = ''.join(alphabet)
    converter = strLabelConverter(alphabet)

    CRNN_model = CRNN.CRNN(CRNN_cfg.imgH, 3, len(alphabet) + 1, CRNN_cfg.hidden_size).to(device)
    CRNN_model.load_state_dict(torch.load(os.path.join(
        CRNN_cfg.pth_path,'CRNN_epoch1005_loss1073_acc0.8327645051194539.pth'),
        map_location=device))
    # CRNN结束-------------------------------------------------------------

    CRNN_ans={}
    for key,img_file in img_files.items():
        CRNN_ans[key]=[]
        EAST_ans=eval_EAST(EAST_model,img_file,EAST_cfg.result_path)
        img, boxes = EAST_ans[0]
        var1=np.array(boxes)[:,:-1]
        boxes=boxes_process(var1,img.size)
        temp_path=os.path.join(EAST_cfg.result_path,key)
        i=0
        for items in boxes:
            for box in items:
                crop_img(box,img,os.path.join(temp_path,'{}.png'.format(i)))
                var1=eval_CRNN(CRNN_model,os.path.join(temp_path,'{}.png'.format(i)),
                    CRNN_cfg.result_path,converter)
                CRNN_ans[key].append(var1)
                i+=1
        # 删除文件夹
        # shutil.rmtree(temp_path)
    
    
    # with open('jiazhao.pk','wb') as file:
    #     pickle.dump(np.array(CRNN_ans['jiazhao']),file)
    # with open('xingchengka.pk','wb') as file:
    #     pickle.dump(np.array(CRNN_ans['xingchengka']),file)
    # with open('hesuan.pk','wb') as file:
    #     pickle.dump(np.array(CRNN_ans['hesuan']),file)
    
    # CRNN_ans={}
    # with open('jiazhao.pk', 'rb') as file_1:
    #     CRNN_ans['jiazhao'] = pickle.load(file_1)
    # with open('xingchengka.pk', 'rb') as file_1:
    #     CRNN_ans['xingchengka'] = pickle.load(file_1)
    # with open('hesuan.pk', 'rb') as file_1:
    #     CRNN_ans['hesuan'] = pickle.load(file_1)
    CRNN_dict=text_process(CRNN_ans)
    return CRNN_dict


if __name__=='__main__':
    OCR(os.path.join(EAST_cfg.dataset_path,'demo/jiazhao.jpg'),
        os.path.join(EAST_cfg.dataset_path,'demo/hesuan.jpg'),
        os.path.join(EAST_cfg.dataset_path,'demo/xingchengka.jpg'))
