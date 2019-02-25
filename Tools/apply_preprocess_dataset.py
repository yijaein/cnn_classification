import os

import cv2

from Works.data_augmentation import apply_clahe


def norm_path(path):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def image_files(path):
    path = norm_path(path)

    img_list = list()
    for (root, dirs, files) in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in ['.png', '.jpg']:
                continue
            file_path = os.path.join(root, file)
            img_list.append(file_path)
    return img_list


def main(src_path, dst_path):
    src_path = norm_path(src_path)
    dst_path = norm_path(dst_path)

    img_files = image_files(src_path)
    for file in img_files:
        # read image file
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        # apply denoise
        # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        
        # apply adaptive histogram equalization
        img = apply_clahe(img, 2.0)

        # following more preprocess
        # img = apply...()

        dst = file.replace(src_path, dst_path)
        dst_file_path = os.path.split(dst)[0]
        if not os.path.exists(dst_file_path):
            os.makedirs(dst_file_path)
        print(dst)
        cv2.imwrite(dst, img)


if __name__ == '__main__':
    src_path = '/home/bong07/data/US_isangmi_folder'
    dst_path = '/home/bong07/data/US_isangmi_folder_equlHist'

    main(src_path, dst_path)
