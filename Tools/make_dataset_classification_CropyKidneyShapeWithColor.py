import os

import cv2
import numpy as np
from tqdm import tqdm

from Tools.crop_by_seg import calculate_angle, rotate_bound, find_bounding_square
from Tools.crop_by_seg import preproess_image
from Tools.make_dataset_classification_CropyKidneyShape import kidney_file_set


import matplotlib.pyplot as plt

# 신장과 비신장 분류를 위한 신장 "모양" 크롭 데이터셋 생성


def norm_path(path):
    path = os.path.normpath(path)
    path = os.path.normcase(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def file_dict(path):
    path = norm_path(path)

    d = dict()
    for (root, dirs, files) in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in ['.jpg', '.png']:
                continue
            d[name] = os.path.join(root, file)
    return d


def main(us_path, mask_path, save_path, padding_size=20, train=True):
    us_path = norm_path(us_path)
    mask_path = norm_path(mask_path)
    save_path = norm_path(save_path)


    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, 'normal'))
        os.makedirs(os.path.join(save_path, 'AKI'))
        os.makedirs(os.path.join(save_path, 'CKD'))

    us_name = file_dict(us_path)
    mask_name = file_dict(mask_path)



    for name_order in tqdm(mask_name.keys()):
        print('name_order:', name_order)
        if '#' in name_order:
            name, order = name_order.split('#')
            print('name:', name)
            print('order:', order)
        else:
            name = name_order
            continue
            print('no has: #')





        mask_img = cv2.imread(mask_name[name_order], cv2.IMREAD_GRAYSCALE)
        us_img = cv2.imread(us_name[name], cv2.IMREAD_GRAYSCALE)

        # Add black padding to input image and seg image
        mask_img = cv2.copyMakeBorder(mask_img, padding_size, padding_size, padding_size, padding_size, 0)
        us_img = cv2.copyMakeBorder(us_img, padding_size, padding_size, padding_size, padding_size, 0)

        # rotate image
        angle = calculate_angle(mask_img)
        mask_img = rotate_bound(mask_img, angle)
        us_img = rotate_bound(us_img, angle)

        # get white pixel bounding box
        x, y, w, h = find_bounding_square(mask_img, padding=padding_size)

        # crop bounding
        mask_img = mask_img[int(y):int(y + h), int(x):int(x + w)]
        us_img = us_img[int(y):int(y + h), int(x):int(x + w)]

        # mask image
        shape_img = np.where(mask_img == 255, us_img, 0)

        result_img = preproess_image(shape_img, mask_img)

        # has kidney?
        if order == '0':
            if train:
                has_kidney = name in seg_name.keys()
            else:
                has_kidney = name in kidney_set
        else:
            has_kidney = False

        dst_path = os.path.join(save_path, 'kidney' if has_kidney else 'non-kidney')
        dst_file = os.path.join(dst_path, name_order + '.png')

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # print(dst_file)
        cv2.imwrite(dst_file, result_img)


if __name__ == '__main__':
    us_path = norm_path('/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded')
    kidney_list_csv = norm_path('/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/기기별 정제 영상 리스트(전체)_3차.csv')
    padding_size = 20

    mask_path = norm_path(
        '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/results_us3_mrcnn_all/all/segKidney_MRCNN_ALL')
    save_path = norm_path('~/data/KorNK/1200/CropKidneyShapeWithColor1')
    main(us_path, mask_path, kidney_list_csv, save_path, padding_size=20, train=True)
