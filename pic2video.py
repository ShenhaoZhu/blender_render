import os
import cv2
import numpy as np


def get_idx(filename):
    return int(filename.split('_')[0])


def pic2video(path, re=False):
    # idx = path.split('_')[-1]
    filelist = [f for f in os.listdir(path) if f.endswith('png')]
    filelist_re = [f for f in os.listdir(path) if f.endswith('png')]
    # filelist.remove('model')
    filelist.sort()
    filelist_re.sort(key=get_idx, reverse=True)
    if re:
        filelist = filelist + filelist_re
    fps = 24 #视频每秒24帧
    size = (512, 512) #需要转为视频的图片的尺寸
    #可以使用cv2.resize()进行修改

    video = cv2.VideoWriter(os.path.join(path, f'../output.mp4'), cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, size)
    #视频保存在当前目录下

    for item in filelist:
        if item.endswith('.png'):
        #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            item = os.path.join(path, item)
            img = cv2.imread(item)
            video.write(img)

    # for item in reversed(filelist):
    #     if item.endswith('.png'):
    #     #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
    #         item = os.path.join(path, item)
    #         img = cv2.imread(item)
    #         video.write(img)

    video.release()

if __name__ == '__main__':
    pic2video('./test_trans_ill_cycles/images')