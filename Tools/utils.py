import os

import cv2
import numpy as np

'''
경로 정규화
(틸트~ 확장, 절대경로 등등)
'''


def norm_path(path, makedirs=False):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if makedirs and not os.path.exists(path):
        os.makedirs(path)
        print('makedirs:, path')
    return path


'''
이미지 파일들의 전체 경로를 포함하는 리스트 구성
recursive = True : 하위 디렉토리 까지 포함
예)
['/path/fileA.png',
 '/path/fileB.png',
 '/path/fileC.png',
 ...]  
'''


def image_list(path, exts=['.png', '.jpg'], recursive=True, followlinks=True):
    path = norm_path(path)

    l = list()
    if recursive:
        for (root, dirs, files) in os.walk(path, followlinks=followlinks):
            for file in files:
                name, ext = os.path.splitext(file)

                if ext.lower() not in exts:
                    continue

                l.append(os.path.join(root, file))
    else:
        for fileDir in os.listdir(path):
            if os.path.isfile(os.path.join(path, fileDir)):
                file = fileDir
            else:
                continue

            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue

            l.append(os.path.join(path, file))
    return l


'''
이미지 파일들의 전체 경로를 포함하는 딕셔너리를 구성, key는 파일 이름 부분(확장자 제외)
recursive = True : 하위 디렉토리 까지 포함
예)
{'fileA': '/path/fileA.png',
 'fileB': '/path/fileB.png',
 'fileC': '/path/fileC.png',
 ...}
'''


def image_dict(path, exts=['.png', '.jpg'], recursive=True, key=None, followlinks=True):
    path = norm_path(path)

    if key == None:
        key = lambda p: os.path.splitext(os.path.split(p)[-1])[0]

    d = dict()
    if recursive:
        for (root, dirs, files) in os.walk(path, followlinks=followlinks):
            for file in files:
                name, ext = os.path.splitext(file)

                if ext.lower() not in exts:
                    continue

                full_path = os.path.join(root, file)
                d[key(full_path)] = full_path
    else:
        for fileDir in os.listdir(path):
            if os.path.isfile(os.path.join(path, fileDir)):
                file = fileDir
            else:
                continue

            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue

            full_path = os.path.join(path, file)
            d[key(full_path)] = full_path
    return d


'''
파일 경로를 각 부분별(경로,이름,확장자)로 분리
예)
('~/path', 'fileName', 'fileExt')
'''


def split_path(file_path):
    path, name_ext = os.path.split(file_path)
    name, ext = os.path.splitext(name_ext)

    return path, name, ext


'''
원본 이미지의 세그먼테이션 영역에 붉게 채운다 
'''


def overlap_segmentation(img_ori_color, img_seg_gray, weight=10):
    # make new seg image in red
    img_seg_color = np.zeros(img_ori_color.shape, dtype=np.uint8)
    img_seg_color[:, :, 2] = img_seg_gray

    img_overlap = cv2.addWeighted(img_ori_color, float(100 - weight) * 0.01, img_seg_color, float(weight) * 0.01, 0)

    return img_overlap
