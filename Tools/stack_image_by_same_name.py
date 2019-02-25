import os

import cv2
import numpy as np


def norm_path(path):
    path = os.path.expanduser(path)
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.abspath(path)
    return path


def main(path_list, result_path, center_crop_ratio=1.0):
    path_list = [norm_path(path) for path in path_list]

    file_dict_list = list()
    for path in path_list:
        file_dict = dict()
        for (root, dirs, files) in os.walk(path):
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() not in ['.jpg', '.png']:
                    continue
                file_dict[name] = os.path.join(root, file)
        file_dict_list.append(file_dict)
        print(len(file_dict.keys()), path)

    for key in file_dict_list[0].keys():
        img_path_list = list()
        for file_dict in file_dict_list:
            if key in file_dict.keys():
                img_path_list.append(file_dict[key])

        if len(img_path_list) != len(file_dict_list):
            continue

        img_list = list()
        for img_path in img_path_list:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # center_crop
            h, w, = img.shape[:2]
            height, width = int(h * center_crop_ratio), int(w * center_crop_ratio)
            off_x, off_y = (w - width) // 2, (h - height) // 2
            img = img[off_y: height + off_y, off_x: width + off_x]

            img_list.append(img)

        img_stack = np.hstack(img_list)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file = os.path.join(result_path, key + '.png')

        print(file)
        cv2.imwrite(file, img_stack)


if __name__ == '__main__':
    path_list = ['/media/bong07/895GB/data/yonsei/results_us3_mrcnn/kidney_original/20181106T091512',
                 '/media/bong07/895GB/data/yonsei/results_us3_mrcnn/kidney_equalhist/20181107T093141',
                 '/media/bong07/895GB/data/yonsei/results_us3_mrcnn/kidney_denoise/20181107T094707']
    result_path = '/media/bong07/895GB/data/yonsei/results_us3_mrcnn/merge_ori_hist_denoise'

    main(path_list, result_path, center_crop_ratio=0.6)
