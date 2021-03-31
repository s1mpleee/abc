

import os
from PIL import Image
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

imgPath = '/media/adas/File/wdx/MOT/MCMOT1/src/data/didi/train_annotation/gt/0007/100/85/images/'
videoPath = '/media/adas/File/yanwu/nn1.mp4'
 
images = os.listdir(imgPath)
images.sort()
print(images)
fps = 25  # 每秒25帧数
    # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
fourcc = VideoWriter_fourcc(*"mp4v")
 
image = Image.open(imgPath + images[0])
videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
for im_name in range(len(images)):
    frame = cv2.imread(imgPath + str(im_name) + '.jpg')  # 这里的路径只能是英文路径
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
    print(im_name)
    videoWriter.write(frame)
print("图片转视频结束！")
videoWriter.release()
cv2.destroyAllWindows()
