import torch
import os


from model import CRNN
from model.EAST import EAST
from model import VGGface
from config import VGGface_config as face_cfg
from config import CRNN_config as CRNN_cfg
from config import EAST_config as EAST_cfg
from tool.tool import *
from eval import eval_CRNN,eval_EAST,eval_VGGface



#整合EAST和CRNN，完成文字识别
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def OCR(jiazhao_img_file,hesuan_img_file,xingchengka_img_file):
    r'''识别图片文字
        jiazhao_img_file: 驾照图片路径
        hesuan_img_file: 核酸图片路径
        xingchengka_img_file: 行程卡图片路径
        输出: 返回一个字典. 格式为: { 'color':[1: 绿色, 2: 黄色, 3: 红色]
            '途径城市':'', '核酸检测时间':'',  '核酸检测机构':''}
    '''
    
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
        EAST_ans=eval_EAST(EAST_model,img_file,device,EAST_cfg.result_path)
        img, boxes = EAST_ans[0]
        var1=np.array(boxes)[:,:-1]
        boxes=boxes_process(var1,img.size)
        temp_path=os.path.join(EAST_cfg.result_path,key)
        i=0
        for items in boxes:
            for box in items:
                crop_img(box,img,os.path.join(temp_path,'{}.png'.format(i)))
                var1=eval_CRNN(CRNN_model,os.path.join(temp_path,'{}.png'.format(i)),converter,device)
                CRNN_ans[key].append(var1)
                i+=1
        # 删除文件夹
        shutil.rmtree(temp_path)
    
    
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
    CRNN_ans=text_process(CRNN_ans)
    return CRNN_ans


def face_detection(video_file,img_file):
    r'''人脸检测, 根据数据库中的图片和上传的视频检测是否为同一个人脸
        video_file: 视频文件路径
        img_file: 图片文件路径
        返回bool值, true代表人脸匹配成功, 反之失败
    '''
    confidence=face_cfg.euc
    model=VGGface.net
    model.load_state_dict(torch.load(
        os.path.join(face_cfg.pth_path,'net_parameter.pth'), map_location=device),
        strict=False)
    sample_path=os.path.join(face_cfg.dataset_path,'temp')
    sample_pics_by_video(video_file,sample_path,frame_step=2)
    sample_pic_file=os.listdir(sample_path)
    # 一个视频采样n张图片，当有0.8*n张图片大于置信阈值时则认为判断成功
    cnt=0
    for sample_file in sample_pic_file:
        ans=eval_VGGface(model,os.path.join(sample_path,sample_file),img_file,device)
        if ans < confidence:
            cnt+=1
    # 删除冗余
    shutil.rmtree(sample_path)
    return cnt>0.8*len(sample_pic_file)


if __name__=='__main__':
    OCR(os.path.join(EAST_cfg.dataset_path,'demo/jiazhao.jpg'),
        os.path.join(EAST_cfg.dataset_path,'demo/hesuan.jpg'),
        os.path.join(EAST_cfg.dataset_path,'demo/xingchengka.jpg'))

    # face_detection(os.path.join(face_cfg.dataset_path,'demo/video/yangwenchen.mp4'),
    #     os.path.join(face_cfg.dataset_path,'demo/pic/qinhaojun.png'))
