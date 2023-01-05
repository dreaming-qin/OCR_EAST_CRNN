# 要求

**人证比对及相关信息识别系统设计与实现**
在某些应用场景下，为提高效率、实现快速同行，设计一个信息识别与照片比对系统，完成:

1. 通过摄像头采集到人脸，并与上传的驾驶证照片进行比对；
2. 识别驾照上的文字信息、上传的核酸检测情况截图信息、大数据行程信息截图信息，并确定关键信息，如姓名是否一致，是否满足安全通行条件。

# 想法

第一个需求使用人证对比模型，使用VGGFace实现

第二个需求使用OCR获得文字然后进行文本过滤, 使用EAST（文字检测）+CRNN（文字识别）

# 运行方式

启动firstPage.py即可

# 结果

OCR做得不理想，人证对比准确率较高

## 人证对比

上传同一个人的照片、视频时

![image-20230105174913960](https://raw.githubusercontent.com/dreaming-qin/md_img/master/img/2023-01-05/20230105-17-49-16.png)

![image-20230105174923105](https://raw.githubusercontent.com/dreaming-qin/md_img/master/img/2023-01-05/20230105-17-49-26.png)

上传不同人的照片、视频时

![image-20230105174949097](https://raw.githubusercontent.com/dreaming-qin/md_img/master/img/2023-01-05/20230105-17-49-53.png)

![image-20230105175210895](https://raw.githubusercontent.com/dreaming-qin/md_img/master/img/2023-01-05/20230105-17-52-13.png)

## OCR

上传驾照、核酸、行程卡，得到的结果

![image-20230105175250021](https://raw.githubusercontent.com/dreaming-qin/md_img/master/img/2023-01-05/20230105-17-52-52.png)

![image-20230105174559221](https://raw.githubusercontent.com/dreaming-qin/md_img/master/img/2023-01-05/20230105-17-46-01.png)

