import collections
import csv
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def norm_path(path):
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    path = os.path.normcase(path)
    path = os.path.abspath(path)
    return path


def replaceBasePath(file_path, src_path, dst_path):
    file_path = norm_path(file_path)
    src_path = norm_path(src_path)
    dst_path = norm_path(dst_path)

    new_path = file_path.replace(src_path, dst_path)
    return new_path


def name_list(path, exts=['.png', '.jpg']):
    path = norm_path(path)
    name_list = list()
    for (root, dirs, files) in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue
            name_list.append(name)
    return name_list


def name_dict(path, exts=['.png', '.jpg']):
    path = norm_path(path)
    name_dict = dict()
    for (root, dirs, files) in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue
            name_dict[name] = os.path.join(root, file)
    return name_dict


'''
dicom 비식별화 프로그램에서 생성하는 csv 파일 파서
dicom_csv header: File, Manufacturer, PhysicalUnitsXDirection, PhysicalDeltaX, PhysicalUnitsYDirection, PhysicalDeltaY

dicom_size = read_dicom_size_csv(csv_file)
print(dicom_size[Name]['Manufacturer'])
'''


def read_dicom_pixel_size_csv(csv_file):
    with open(csv_file) as csvfile:
        csv_reader = csv.DictReader(csvfile)

        # key name = first column name (dicom file name)
        key_name = csv_reader.fieldnames[0]
        dicom_size_dict = dict()
        for line in csv_reader:
            dicom_file = line[key_name]
            # strip ext of file name
            dicom_name, ext = os.path.splitext(dicom_file)
            dicom_size_dict[dicom_name] = line

    return dicom_size_dict


'''
진단 엑셀 파일로 부터 만들어진 csv 파일 파서
diagnosis_csv header: Date, File, RecoredPatientID, RealPatientID, AccNo, Diagnosis, Excluded

png 파일들을 폴더(1건) 형태로 구성할때는 폴더 이름을 AnnNo를 사용해야 중복되지 않음 
'''


def read_diagnosis_csv(csv_file, include_excluded=False, only_excluded=False):
    with open(csv_file) as csvfile:
        csv_reader = csv.DictReader(csvfile)

        # key name = first column name (dicom file name)
        key_name = csv_reader.fieldnames[1]
        dicom_size_dict = dict()
        for line in csv_reader:

            if only_excluded == True and line['Excluded'] == '':
                continue

            if include_excluded == False and line['Excluded'] != '':
                continue

            dicom_file = line[key_name]

            # strip ext of file name
            dicom_name, ext = os.path.splitext(dicom_file)
            dicom_size_dict[dicom_name] = line

    return dicom_size_dict


'''
이미지를 새로운 픽셀 축척에 따라 리사이즈
'''


def resize_physical_unit(img, src_pixel_physical_size, dst_pixel_physical_size):
    # order width, height
    if not isinstance(src_pixel_physical_size, collections.Iterable):
        src_pixel_physical_size = (src_pixel_physical_size, src_pixel_physical_size)

    if not isinstance(dst_pixel_physical_size, collections.Iterable):
        dst_pixel_physical_size = (dst_pixel_physical_size, dst_pixel_physical_size)

    ratio_width = src_pixel_physical_size[0] / dst_pixel_physical_size[0]
    ratio_height = src_pixel_physical_size[1] / dst_pixel_physical_size[1]

    img = cv2.resize(img, None, fx=ratio_width, fy=ratio_height)
    return img


'''
이미지를 정해진 축적과 이미지 크기로 패딩을 넣어 리사이즈

img_file = './img_1119_0.2641970.jpg'
img = cv2.imread(img_file)
resize_img = resize_physical_with_pading(img, 0.02641970, (320, 320), dst_cm=(15, 15)) # 15cm x 15cm image(320px x 320px)
resize_img = resize_physical_with_pading(img, 0.02641970, (320, 320), dst_cm=0.035) # (320 x 0.035)cm x (320 x 0.035)cm image
'''


def resize_physical_with_pading(img, src_pixel_physical_size, dst_size, dst_pixel_physical_size=None, dst_cm=None):
    if not isinstance(dst_size, collections.Iterable):
        dst_size = (dst_size, dst_size)

    if dst_pixel_physical_size is None:
        if not isinstance(dst_cm, collections.Iterable):
            dst_cm = (dst_cm, dst_cm)
        dst_pixel_physical_size = dst_cm[0] / dst_size[0], dst_cm[1] / dst_size[1]

    # resize
    img = resize_physical_unit(img, src_pixel_physical_size, dst_pixel_physical_size)
    # img_depth = img.shape[2]

    # crop
    crop_size = min(img.shape[1], dst_size[0]), min(img.shape[0], dst_size[1])
    crop_offset = (img.shape[1] - crop_size[0]) // 2, (img.shape[0] - crop_size[1]) // 2
    crop = img[crop_offset[1]:crop_offset[1] + crop_size[1], crop_offset[0]:crop_offset[0] + crop_size[0]]

    # paste
    # canvas = np.zeros([dst_size[1], dst_size[0], img_depth], dtype=img.dtype)
    canvas = np.zeros([dst_size[1], dst_size[0]], dtype=img.dtype)
    print('canvas.shape', canvas.shape)
    paste_offset = (dst_size[0] - crop_size[0]) // 2, (dst_size[1] - crop_size[1]) // 2,
    print('paste_offset', paste_offset)
    canvas[
    paste_offset[1]: paste_offset[1] + crop_size[1],
    paste_offset[0]: paste_offset[0] + crop_size[0]] = crop[:crop_size[1], :crop_size[0]]

    return canvas


'''
src_path 내의 이미지들을 일괄적으로 동일한 축적과 이미지 크기로 변경하여 저장
dst_path에 저장시에 동일 디렉토리 구조를 유지함
'''


def convert_resize_physical(src_path, dst_path, dicom_pixel_size_csv_path, size_px, size_cm):
    src_path = norm_path(src_path)
    dst_path = norm_path(dst_path)

    # gathering image files
    file_list = list()
    for (root, dirs, files) in os.walk(src_path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in ['.png', '.jpg']:
                continue
            file_list.append(os.path.join(root, file))

    dicom_size_dict = read_dicom_pixel_size_csv(dicom_pixel_size_csv_path)

    # convert image
    cnt_success = 0
    cnt_fail = 0
    for file in file_list:
        _, name_ext = os.path.split(file)
        name, ext = os.path.splitext(name_ext)

        if name not in dicom_size_dict.keys():
            print('not found dicom info', name)
            cnt_fail += 1
            continue

        # File = dicom_size_dict[name]['File']
        # Manufacturer = dicom_size_dict[name]['Manufacturer']
        PhysicalUnitsXDirection = dicom_size_dict[name]['PhysicalUnitsXDirection']
        PhysicalDeltaX = dicom_size_dict[name]['PhysicalDeltaX']
        PhysicalUnitsYDirection = dicom_size_dict[name]['PhysicalUnitsYDirection']
        PhysicalDeltaY = dicom_size_dict[name]['PhysicalDeltaY']

        if 'None' in [PhysicalUnitsXDirection, PhysicalDeltaX, PhysicalUnitsYDirection, PhysicalDeltaY]:
            print('dicom info is None', name)
            cnt_fail += 1
            continue

        PhysicalUnitsXDirection = int(PhysicalUnitsXDirection)
        PhysicalDeltaX = float(PhysicalDeltaX)
        PhysicalUnitsYDirection = int(PhysicalUnitsYDirection)
        PhysicalDeltaY = float(PhysicalDeltaY)

        img = cv2.imread(file)
        resize_img = resize_physical_with_pading(img, (PhysicalDeltaX, PhysicalDeltaY), size_px, dst_cm=size_cm)
        dst = replaceBasePath(file, src_path, dst_path)

        dst_file_path, _ = os.path.split(dst)
        if not os.path.exists(dst_file_path):
            os.makedirs(dst_file_path)

        cv2.imwrite(dst, resize_img)
        cnt_success += 1

    print('cnt_success', cnt_success)
    print('cnt_fail', cnt_fail)


'''
박스 포인트(4지점)이 주어지면 박스의 폭(긴쪽)과 높이(짧은쪽)을 반환
'''


def rect_wh(boxPoints):
    p0, p1, p2, p3 = boxPoints
    X = 0
    Y = 1
    dist01 = math.sqrt(math.pow(abs(p0[X] - p1[X]), 2) + math.pow(abs(p0[Y] - p1[Y]), 2))
    dist03 = math.sqrt(math.pow(abs(p0[X] - p3[X]), 2) + math.pow(abs(p0[Y] - p3[Y]), 2))

    # return long dist, short dist
    return (dist01, dist03) if dist01 > dist03 else (dist03, dist01)


'''
세그먼트 이미지의 분할 영역을 감싸는 가장 작은 직사각형의 박스포인트를 반환

data_info = list()
pixel_size = 15 / 234
for file_path in data_list:
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img, longd, shortd = object_wh(img)
    long_cm, short_cm = longd * pixel_size, shortd * pixel_size
    info = (file_path, long_cm , short_cm)
    data_info.append(info)
'''


def object_wh(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    # draw point
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    point_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
    for idx, (x, y) in enumerate(box):
        cv2.circle(img, (x, y), 5, point_color[idx], -1)

    # draw box
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

    # plt.imshow(img)
    # plt.show()

    d0, d1 = rect_wh(box)

    return img, d0, d1


'''
지정한 이미지들의 정보를 출력

output: 기기, 폴더, 파일, 진단, 긴쪽, 짧은쪽, 폴더내 신장 순서

diagnosis_csv header: Date, File, RecoredPatientID, RealPatientID, AccNo, Diagnosis, Excluded
dicom_csv     header: File, Manufacturer, PhysicalUnitsXDirection, PhysicalDeltaX, PhysicalUnitsYDirection, PhysicalDeltaY
'''


def get_kidney_info(image_name_list, seg_path, diagnosis_csv, dicom_csv):
    diagnosis_info = read_diagnosis_csv(diagnosis_csv)
    dicom_info = read_dicom_pixel_size_csv(dicom_csv)

    folder_dict = dict()
    for name in image_name_list:
        folder = diagnosis_info[name]['AccNo']
        if folder not in folder_dict.keys():
            folder_dict[folder] = list()
        folder_dict[folder].append(name)

    # gather seg files
    seg_dict = dict()
    for (root, dirs, files) in os.walk(seg_path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in ['.png', '.jpg']:
                continue
            seg_dict[name] = os.path.join(root, file)

    kidney_info = dict()
    for folder in folder_dict.keys():
        info_list = list()
        for name in folder_dict[folder]:
            if name not in seg_dict.keys():
                print('not found seg', name)
                continue
            if name not in dicom_info.keys():
                print('not found dicom_info', name)
                continue

            Manufacturer = dicom_info[name]['Manufacturer']
            Folder = folder
            File = name
            Diagnosis = diagnosis_info[name]['Diagnosis']

            # compute Long cm, Short cm
            seg_img = cv2.imread(seg_dict[name], cv2.IMREAD_GRAYSCALE)
            _, long_px, short_px = object_wh(seg_img)
            PhysicalDeltaX, PhysicalDeltaY = dicom_info[name]['PhysicalDeltaX'], dicom_info[name]['PhysicalDeltaY']
            PhysicalDeltaX = float(PhysicalDeltaX) if PhysicalDeltaX != 'None' else 0.0
            PhysicalDeltaY = float(PhysicalDeltaY) if PhysicalDeltaY != 'None' else 0.0
            LongCM, ShortCM = long_px * PhysicalDeltaX, short_px * PhysicalDeltaY

            info = [Manufacturer, Folder, File, Diagnosis, LongCM, ShortCM]
            info_list.append(info)

        # LongCM index = 4
        info_list = sorted(info_list, key=lambda x: x[4], reverse=True)
        for idx in range(len(info_list)):
            OrderLong = idx + 1
            info_list[idx].append(OrderLong)

        # File index = 2
        for info in info_list:
            name = info[2]
            kidney_info[name] = info

    return kidney_info


def show_kidney_size_distribution_graph_per_diagnosis(image_path, seg_path, diagnosis_csv, dicom_csv):
    def show_plt(xs, ys, color='black', title='', xlabel='long cm', ylabel='short cm', ylim=(0, 16), xlim=(0, 16),
                 show=True):
        # point size
        s = 0.5
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        plt.grid(True, lw=1, ls='--', c='0.5')
        plt.scatter(xs, ys, color=color, s=s)
        if show:
            plt.show()

    kidney_info = get_kidney_info(image_path, seg_path, diagnosis_csv, dicom_csv)

    xys_ckd = list()
    xys_aki = list()
    xys_nor = list()
    xys_all = list()

    # use top1~2 of image size
    n_top = 1

    for key, (Manufacturer, Folder, File, Diagnosis, LongCM, ShortCM, OrderLong) in kidney_info.items():
        # pass small kidney
        if n_top < OrderLong:
            continue

        Diagnosis = Diagnosis.lower()
        xys_all.append([LongCM, ShortCM])
        if Diagnosis == 'ckd':
            xys_ckd.append([LongCM, ShortCM])
        elif Diagnosis == 'aki':
            xys_aki.append([LongCM, ShortCM])
        elif Diagnosis == 'normal':
            xys_nor.append([LongCM, ShortCM])
        else:
            print('unknow({}) diagnosis {}'.format(Diagnosis, File))
            continue

    xys_ckd = np.array(xys_ckd)
    xys_aki = np.array(xys_aki)
    xys_nor = np.array(xys_nor)
    xys_all = np.array(xys_all)

    # Long CM = 0, Short CM = 1
    X = 0
    Y = 1

    # print mean, stddev
    print('ckd mean:{:.5}, std:{:.5}'.format(np.mean(xys_ckd[:, X]), np.std(xys_ckd[:, X])))
    print('aki mean:{:.5}, std:{:.5}'.format(np.mean(xys_aki[:, X]), np.std(xys_aki[:, X])))
    print('nor mean:{:.5}, std:{:.5}'.format(np.mean(xys_nor[:, X]), np.std(xys_nor[:, X])))
    print('all mean:{:.5}, std:{:.5}'.format(np.mean(xys_all[:, X]), np.std(xys_all[:, X])))

    print('area')
    print('ckd mean:{:.5}, std:{:.5}'.format(np.mean(xys_ckd[:, X] * xys_ckd[:, Y]),
                                             np.std(xys_ckd[:, X] * xys_ckd[:, Y])))
    print('aki mean:{:.5}, std:{:.5}'.format(np.mean(xys_aki[:, X] * xys_aki[:, Y]),
                                             np.std(xys_aki[:, X] * xys_aki[:, Y])))
    print('nor mean:{:.5}, std:{:.5}'.format(np.mean(xys_nor[:, X] * xys_nor[:, Y]),
                                             np.std(xys_nor[:, X] * xys_nor[:, Y])))
    print('all mean:{:.5}, std:{:.5}'.format(np.mean(xys_all[:, X] * xys_all[:, Y]),
                                             np.std(xys_all[:, X] * xys_all[:, Y])))

    # show graph
    show_plt(title='ckd', xs=xys_ckd[:, X], ys=xys_ckd[:, Y], color='red')
    show_plt(title='aki', xs=xys_aki[:, X], ys=xys_aki[:, Y], color='green')
    show_plt(title='normal', xs=xys_nor[:, X], ys=xys_nor[:, Y], color='blue')

    show_plt(title='ckd', xs=xys_ckd[:, X], ys=xys_ckd[:, Y], color='red', show=False)
    show_plt(title='aki', xs=xys_aki[:, X], ys=xys_aki[:, Y], color='green', show=False)
    show_plt(title='all', xs=xys_nor[:, X], ys=xys_nor[:, Y], color='blue', show=True)
