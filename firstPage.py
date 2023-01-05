import sys
import os

import PySimpleGUI as sg
import front_end.sources as ss
from shutil import copy
from shutil import rmtree
from time import sleep
from solution import OCR,face_detection



# 步骤如下：
# 1.首先建立每个页面各个元素
# 2.把每个页面构成框架
# 3.设置每个页面的可见属性
# 4.当点击发生时，设置对应的事件反应
# 5.快速连点击预防措施！！防止系统卡bug

# 导入图片数据
# image1 = ss.image_cc   # 人证比对系统
image_2 = ss.image_wenhao
image_lvma = ss.image_lvma

# 在此处修改temp路径
path_temp = "./temp"

# 首页
layout_FirstPage = [
    [sg.T("人证比对系统", size=(30, 1),
          font=("宋体", 30), justification="center",
          border_width="4", background_color="black")],

    # [sg.T("请选择你需要的操作：")],
    # [sg.B("人证比对", image_data=image1)], # 按钮里的文字会自动居中
    [sg.T("                                           "), sg.B(
        "人脸比对", font=("宋体", 20), button_color=("black", None), image_filename=r"./front_end/pic/人证2.png", image_size=(None, None))],
    [sg.T("                                           "),
     sg.B("文字匹配", font=("宋体", 20), button_color=("black", None),  image_filename=r"./front_end/pic/文字识别.png")],
    [sg.T("                                      "), sg.B("退出系统", size=(30, 3))],
]

# 人证比对页面
layout_Ctcp = [
    [sg.T("帮助"), sg.B(" ", image_data=image_2, key="help")],
    [sg.FileBrowse(button_text="上传视频", target='-MP4-', file_types=(("ALL Files", "mp4"),), key="uploadvideo", size=(40, 15)),
     sg.FileBrowse(button_text="上传图片", target='-PNG-', file_types=(("ALL Files", "*.png *.jpg"),), key="uploadcard1", size=(40, 15))],
    [sg.T("视频路径"), sg.In(key="-MP4-", size=85)],
    [sg.T("图片路径"), sg.In(key="-PNG-", size=85)],
    [sg.B("核对", key="comparision", size=(10, 1)),
     sg.B("返回首页", key="返回首页100", size=(10, 1))]
]

# 人证比对子页面——帮助页面
layout_w = [
    [sg.B("返回", key="back")],
    [sg.T("样例如下：")],
    [sg.Image(filename="./front_end/pic/驾驶证.png", subsample=2),
     sg.Image(filename="./front_end/pic/身份证正面.png", subsample=2),
     sg.Image(filename="./front_end/pic/身份证背面.png", subsample=2),
     sg.Image(filename="./front_end/pic/行程卡.png", subsample=2),

     ]
]

# 人证比对子页面——核对页面
finals_cc = ["正在验证", "验证失败", "验证成功"]
# 设置时间，时间结束验证失败，或者在规定时间内验证成功
layout_cc = [
    [sg.B(finals_cc[0], key="人脸比对结果", size=(60, 12))],
    [sg.B("返回首页", key="返回首页3"), sg.B("重新上传", key="back2Ctcp")]
]


# 文字匹配页面
layout_TM = [    # Text Matching  文字匹配
    [sg.T("帮助"), sg.B("", image_data=image_2, key="help2")],
    [sg.FileBrowse(button_text="驾照", target='-img1-', file_types=(("ALL Files", "*.png *.jpg"),), key="card2", size=(30, 10)),
        sg.FileBrowse(button_text="核酸证明", target='-img2-',
                      file_types=(("ALL Files", "*.png *.jpg"),), key="hesuan", size=(30, 10)),
        sg.FileBrowse(button_text="行程卡", target='-img3-', file_types=(("ALL Files", "*.png *.jpg"),), key="waycard", size=(30, 10))],
    [sg.T("驾  照"), sg.In(key="-img1-", size=85)],
    [sg.T("核  酸"), sg.In(key="-img2-", size=85)],
    [sg.T("行程卡"), sg.In(key="-img3-", size=85)],
    [sg.B("确定上传", size=(20, 2)), sg.T(' '*70),
     sg.B("返回首页", size=(20, 2), key="返回首页2")]
]

release_ = ["可以通行", "禁止通行"]
layout_final2 = [
    [sg.B("正在验证", key="识别结果")],
    [sg.T(" "*50), sg.B("", key="OCR结果", image_filename=r"./front_end/pic/bai.png", )],
    # [sg.T("姓名："+getname), sg.T(getname,)],
    [sg.B("**", key="途径城市", size=60)],
    [sg.B("**", key="核酸检测时间", size=60)],
    [sg.B("**", key="核酸检测机构", size=60)],
    [sg.B("返回", key="bk2cp"), sg.B("返回首页", key="返回首页4")]
]

# 总页面，通过设置可见属性，来实现页面的切换
layout = [
    [sg.Frame("", key="FirstPage", layout=layout_FirstPage, border_width=0),
     sg.Frame("", key="Ctcp", layout=layout_Ctcp,
              border_width=0, visible=False),
     sg.Frame("", key="wenhao", layout=layout_w,
              border_width=0, visible=False),
     sg.Frame("", key="TM", layout=layout_TM, border_width=0, visible=False),
     sg.Frame("", border_width=0, key="ShowMassages",
              layout=layout_cc, visible=False),
     # sg.Frame("",key = "ma",border_width=0,layout = layout_final1,visible=False),
     sg.Frame(" ", key="xinxi", border_width=0,
              layout=layout_final2, visible=False)
     ]
]
window = sg.Window("人证比对系统", layout)

flag = -1    # 用来指示返回的界面是哪个

while True:
    event, values = window.read()   # 读取窗口的输入

    if event == sg.WINDOW_CLOSED:  # 关闭窗口
        break

    if event == "退出系统":
        break

    if event == "人脸比对":      # 页面跳转
        flag = 0   # 0表示k 从示例返回 会回到 人证比对界面
        print("点击了人脸比对按钮，跳转到人脸比对界面")
        window["Ctcp"].update(visible=True)
        window["FirstPage"].update(visible=False)

    if event in ("文字匹配", "bk2cp"):
        flag = 1    #
        print("用户点击了文字匹配")
        window["TM"].update(visible=True)
        window["FirstPage"].update(visible=False)
        window["xinxi"].update(visible=False)

        # 文字匹配恢复设置
        window['识别结果'].update(text="正在验证")
        window['OCR结果'].update(image_filename=r"./front_end/pic/bai.png")
        window["途径城市"].update(text="**")
        window["核酸检测时间"].update(text="**")
        window["核酸检测机构"].update(text="**")

    if event in ("help", "help2"):
        print("点击了帮助")
        window["Ctcp"].update(visible=False)
        # window["FirstPage"].update(visible=False)
        window["TM"].update(visible=False)
        window["wenhao"].update(visible=True)

    if flag == 0 and event == "back":

        print("返回人脸比对")
        window["Ctcp"].update(visible=True)
        # window["FirstPage"].update(visible=False)
        window["wenhao"].update(visible=False)

    if flag == 1 and event == "back":
        print("返回文字匹配")
        window["TM"].update(visible=True)
        # window["FirstPage"].update(visible=False)
        window["wenhao"].update(visible=False)

    if event == "上传视频":  # 进行文件选择，只显示视频.mp4后缀和文件夹
        # 文件选择之后提取关键帧作为图片显示在按钮上，记录文件路径，刷新页面
        # 同时新建文件夹，将路径对应的文件复制到当前文件夹下
        print('点击了上传视频')

    if event == "上传图片":
        print("点击了上传图片")

    if event == "comparision":  # 人脸比对
        print("点击了比对")
        window["Ctcp"].update(visible=False)
        window["ShowMassages"].update(visible=True)

        mp4str = values["-MP4-"]  # 获取路径
        pngstr = values["-PNG-"]
        print("视频路径: "+mp4str)  # 输出路径
        print("图片路径："+pngstr)
        # 复制文件到temp文件夹
        if os.path.exists(path_temp):
            rmtree(path_temp)
        os.makedirs(path_temp)  # 创建temp
        copy(mp4str, path_temp)
        copy(pngstr, path_temp)
        # 设置弹窗，调用人脸比对函数，弹窗显示结果
        # 首先对路径进行切割，只保留最后的文件名
        video_file = os.path.join(path_temp ,mp4str.split('/')[-1])
        img_file = os.path.join(path_temp ,pngstr.split('/')[-1])

        # sleep(2)  #显示弹窗 正在核对
        [sg.popup("正在验证",
                  title="comp_window",
                  non_blocking=True,  # 自动运行下一行代码
                  auto_close=False,  # 自动关闭弹窗
                  auto_close_duration=2,  # 关闭之前保持的时间
                  )]

        # final_face = True
        final_face = face_detection(video_file,img_file)
        if final_face == True:
            print("显示验证成功")
            window["人脸比对结果"].update(text="验证成功")
        else:
            print("显示验证失败")
            window["人脸比对结果"].update(text="验证失败")

        rmtree(path_temp)
        window["Ctcp"].update(visible=False)
        window["ShowMassages"].update(visible=True)

    if event == "back2Ctcp":
        print("返回上一级")
        window["人脸比对结果"].update(text='正在验证')
        window["Ctcp"].update(visible=True)
        window["ShowMassages"].update(visible=False)

    if event == "确定上传":
        window["TM"].update(visible=False)
        window["xinxi"].update(visible=True)

        # 获取三个路径，建立temp文件夹，复制文件到temp文件夹下，OCR,显示结果,删除文件夹
        jiazhao_str = values["-img1-"]
        hesuan_str = values['-img2-']
        xingchengka_str = values['-img3-']
        print("驾照路径：" + jiazhao_str)
        print("核酸路径：" + hesuan_str)
        print("行程卡路径：" + xingchengka_str)

        if os.path.exists(path_temp):
            rmtree(path_temp)
        os.makedirs(path_temp)

        copy(jiazhao_str, path_temp)
        copy(hesuan_str, path_temp)
        copy(xingchengka_str, path_temp)
        jiazhao_img_file=os.path.join(path_temp ,jiazhao_str.split('/')[-1])
        hesuan_img_file=os.path.join(path_temp ,hesuan_str.split('/')[-1])
        xingchengka_img_file=os.path.join(path_temp ,xingchengka_str.split('/')[-1])
        [sg.popup("正在验证",
                  title="comp_window",
                  non_blocking=True,  # 自动运行下一行代码
                  auto_close=True,  # 自动关闭弹窗
                  auto_close_duration=2,  # 关闭之前保持的时间
                  )]
        final_OCR = OCR(jiazhao_img_file,hesuan_img_file,xingchengka_img_file)
        # final_OCR = {'color': 1, '途径城市': '111',
        #              '核酸检测时间': '2222',  '核酸检测机构': '3333'}
        sleep(2)
        window["识别结果"].update(text="验证成功")
        # 结果的字典不一定全都有,检测他有什么
        mal = ["./front_end/pic/green.png",
            "./front_end/pic/yellow.png","./front_end/pic/red.png"]
        ma_color = final_OCR["color"]-1

        window["OCR结果"].update(image_filename=mal[ma_color])
        if "途径城市" in final_OCR:
            window["途径城市"].update(text="途径城市:"+final_OCR["途径城市"])
        if "核酸检测时间" in final_OCR:
            window["核酸检测时间"].update(text="核酸检测时间:"+final_OCR["核酸检测时间"])
        if "核酸检测机构" in final_OCR:
            window["核酸检测机构"].update(text="核酸检测机构:"+final_OCR["核酸检测机构"])

        rmtree(path_temp)

    if event in ("返回首页0", "返回首页2", "返回首页3", "返回首页4", "返回首页100"):
        print("点击了返回首页")
        # 人脸比对恢复设置
        window["人脸比对结果"].update(text='正在验证')

        # 文字匹配恢复设置
        window['识别结果'].update(text="正在验证")
        window['OCR结果'].update(image_filename=r"./front_end/pic/bai.png")
        window["途径城市"].update(text="**")
        window["核酸检测时间"].update(text="**")
        window["核酸检测机构"].update(text="**")

        window["Ctcp"].update(visible=False)
        window["TM"].update(visible=False)
        window["FirstPage"].update(visible=True)
        window["ShowMassages"].update(visible=False)
        window["xinxi"].update(visible=False)


window.close()
