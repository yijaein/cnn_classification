import os

import cv2
import numpy as np

from Tools.utils import norm_path, image_dict
'''
세그먼테이션이 정확한지 확인하기 위해 원본 초음파 이미지와 세그먼테이션 이미지를 오버랩하여 저장 
'''


def overlap_us_seg(us_path, seg_paths, w=20):
    us_path = norm_path(us_path)
    seg_paths = [norm_path(p) if p else None for p in seg_paths]

    img_us = cv2.imread(us_path, cv2.IMREAD_COLOR)
    # resize
    img_us = cv2.resize(img_us, dsize=(512, 512))

    img_seg_list = []
    for p in seg_paths:
        if p:
            img_seg = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

            img_seg_color = np.zeros(img_us.shape, dtype=np.uint8)
            img_seg_color[:, :, 2] = img_seg
            img_seg_color = cv2.addWeighted(img_us, float(100 - w) * 0.01, img_seg_color, float(w) * 0.01, 0)
        else:
            img_seg_color = np.zeros(img_us.shape, dtype=np.uint8)
        img_seg_list.append(img_seg_color)

    img_stack = np.hstack([img_us] + img_seg_list)
    return img_stack


def make_overlap_us_seg(us_path, seg_paths, save_path, pass_without_seg=True):
    us_path = norm_path(us_path)
    seg_paths = [norm_path(p) for p in seg_paths]
    save_path = norm_path(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    us_dict = image_dict(us_path)
    key = lambda p: os.path.splitext(os.path.split(p)[-1])[0].split('#')[0]
    seg_dict_list = [image_dict(p, key=key) for p in seg_paths]

    for name in us_dict:
        seg_path_list = []

        for seg_dict in seg_dict_list:
            if name in seg_dict:
                seg_path_list.append(seg_dict[name])
            else:
                seg_path_list.append(None)

        if pass_without_seg and None in seg_path_list:
            continue

        img = overlap_us_seg(us_dict[name], seg_path_list)
        save_file = os.path.join(save_path, name + '.png')
        print(save_file)
        cv2.imwrite(save_file, img)


if __name__ == '__main__':
    '''
    seg_paths 는 리스트로 여러 경로를 지정 할 수 있음
    pass_without_seg이 True이면 seg_paths에서 하나라도 찾지 못할 경우 출력 안함, False이면 해당 자리는 비워둠 
    '''

    us_path = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded/val'
    seg_paths = ['/home/bong6/lib/robin_yonsei3/result_us1_model1/seg_result22']
    save_path = '/home/bong6/lib/robin_yonsei3/result_us1_model1/overray2'

    make_overlap_us_seg(us_path, seg_paths, save_path, pass_without_seg=True)
