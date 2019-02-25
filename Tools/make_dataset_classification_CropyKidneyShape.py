import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

from Tools.crop_by_seg import calculate_angle, rotate_bound, find_bounding_square


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


def kidney_file_set(kidney_csv_file):
    kidney_csv_file = norm_path(kidney_csv_file)
    s = set()
    with open(kidney_csv_file) as fkidney:
        csv_reader = csv.DictReader(fkidney)

        # key_name = csv_reader.fieldnames[0]
        key_name = 'File'
        for line in csv_reader:
            s.add(line[key_name])
    return s


def main(seg_path, us_path, mask_path, kidney_list_csv, save_path, padding_size=20, train=True):
    seg_path = norm_path(seg_path)
    us_path = norm_path(us_path)
    mask_path = norm_path(mask_path)
    save_path = norm_path(save_path)
    kidney_list_csv = norm_path(kidney_list_csv)

    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, 'kidney'))
        os.makedirs(os.path.join(save_path, 'non-kidney'))

    seg_name = file_dict(seg_path)
    us_name = file_dict(us_path)
    mask_name = file_dict(mask_path)
    kidney_set = kidney_file_set(kidney_list_csv)

    for name_order in tqdm(mask_name.keys()):
        name, order = name_order.split('#')

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
        cv2.imwrite(dst_file, shape_img)


if __name__ == '__main__':
    seg_path = norm_path('~/data/SegKidney')
    us_path = norm_path('/media/bong07/895GB/data/yonsei/png_classification/once_400+100+1200')
    kidney_list_csv = norm_path('~/data/yonsei/doc/기기별_정제_데이터_영상/기기별 정제 영상 리스트(전체)_3차.csv')
    padding_size = 20

    mask_path = norm_path(
        '~/lib/robin_yonsei/results_us3_mrcnn_1200/kidney/all/SegKidney_MRCNN')
    save_path = norm_path('~/data/KorNK/1200/CropKidneyShape')
    main(seg_path, us_path, mask_path, kidney_list_csv, save_path, padding_size=20, train=True)
