import matplotlib.pyplot as plt
import os
from Tools.dicom_physical_size import get_kidney_info, resize_physical_with_pading, object_wh, norm_path, name_dict, read_dicom_pixel_size_csv, read_diagnosis_csv
import cv2
import numpy as np
import random


def show_plt(xs, ys, color='black', title='', xlabel='long cm', ylabel='short cm', ylim=(0, 3), xlim=(0, 90000),
             show=True):
    # point size
    s = 0.5
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.grid(True, lw=1, ls='--', c='0.5')
    plt.scatter(xs, ys, color=color, s=s)
    if show:
        plt.show()


def main():


    # setting
    src_path = norm_path('/home/bong07/data/CropKidney_made_Machine_isangmi_top2/val')
    # src_path = norm_path('/home/bong07/data/yonsei/doc/기기별_정제_데이터_영상/CropKidney_isangmi_only100400')
    seg_path = norm_path('/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegKidney_v3')
    size_px = 300
    size_cm = 15
    save_csv = norm_path('~/data/yonsei/debug_kidney_size.csv')


    #fix
    diagnosis_csv = norm_path('/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/doc/진단정보/diagnosis_info_400+100+1200.csv')
    dicom_info_csv = norm_path('/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/doc/Dicom정보/dicom_info_100+400+1200.csv')


    src_dict = name_dict(src_path)
    seg_dict = name_dict(seg_path)
    diagnosis_dict = read_diagnosis_csv(diagnosis_csv)
    dicom_dict = read_dicom_pixel_size_csv(dicom_info_csv)


    with open(save_csv, 'wt') as fcsv:

        header = ','.join(['Name', 'Diagnosis', 'Long_px', 'Short_px', 'Long_cm', 'Short_cm', 'Volume'])
        fcsv.write(header + '\n')

        kidney_info = dict()
        for name in src_dict.keys():

            # get Diagnosis
            if name in diagnosis_dict.keys():
                Diagnosis = diagnosis_dict[name]['Diagnosis'].lower()
            else:
                print('not found diagnosis info', name)
                continue

            # get pixel size
            if name in dicom_dict.keys():
                PhysicalUnitsXDirection = dicom_dict[name]['PhysicalUnitsXDirection']
                PhysicalDeltaX = dicom_dict[name]['PhysicalDeltaX']
                PhysicalUnitsYDirection = dicom_dict[name]['PhysicalUnitsYDirection']
                PhysicalDeltaY = dicom_dict[name]['PhysicalDeltaY']

                if 'None' in [PhysicalUnitsXDirection, PhysicalDeltaX, PhysicalUnitsYDirection, PhysicalDeltaY]:
                    print('dicom size is None', name)
                    continue
                else:
                    PhysicalDeltaX = float(PhysicalDeltaX)
                    PhysicalDeltaY = float(PhysicalDeltaY)

            else:
                print('not found dicom info', name)
                continue


            # get kidney Long cm, Short cm
            img_seg = cv2.imread(seg_dict[name], cv2.IMREAD_GRAYSCALE)
            _, long_px, short_px = object_wh(img_seg)
            LongCM, ShortCM = long_px * PhysicalDeltaX, short_px * PhysicalDeltaY

            # resize_physical
            img = cv2.imread(seg_dict[name], cv2.IMREAD_GRAYSCALE)
            resize_img = resize_physical_with_pading(img, (PhysicalDeltaX, PhysicalDeltaY), size_px, dst_cm=size_cm)
            volume = np.count_nonzero(resize_img)

            # print(volume)
            # plt.imshow(img, cmap='gray')
            # plt.show()
            #
            # plt.imshow(resize_img, cmap='gray')
            # plt.show()



            kidney_info[name] = [name, Diagnosis, LongCM, ShortCM, volume]
            print([name, Diagnosis, LongCM, ShortCM, volume])

            line = ','.join([name, Diagnosis, str(long_px), str(short_px), str(LongCM), str(ShortCM), str(volume)])
            fcsv.write(line + '\n')


    xys_ckd = list()
    xys_aki = list()
    xys_nor = list()
    xys_all = list()
    for name in kidney_info.keys():
        _, Diagnosis, LongCM, ShortCM, volume = kidney_info[name]

        dummy = random.random() + 1.0

        xys_all.append([volume, dummy])
        if Diagnosis == 'ckd':
            xys_ckd.append([volume, dummy])
        elif Diagnosis == 'aki':
            xys_aki.append([volume, dummy])
        elif Diagnosis == 'normal':
            xys_nor.append([volume, dummy])
        else:
            print('err Diagnosis', name)
            exit()

    xys_ckd = np.array(xys_ckd)
    xys_aki = np.array(xys_aki)
    xys_nor = np.array(xys_nor)
    xys_all = np.array(xys_all)
    X = 0
    Y = 1


    # show graph
    show_plt(title='ckd', xs=xys_ckd[:, X], ys=xys_ckd[:, Y], color='red')
    show_plt(title='aki', xs=xys_aki[:, X], ys=xys_aki[:, Y], color='green')
    show_plt(title='normal', xs=xys_nor[:, X], ys=xys_nor[:, Y], color='blue')

    show_plt(title='ckd', xs=xys_ckd[:, X], ys=xys_ckd[:, Y], color='red', show=False)
    show_plt(title='aki', xs=xys_aki[:, X], ys=xys_aki[:, Y], color='green', show=False)
    show_plt(title='all', xs=xys_nor[:, X], ys=xys_nor[:, Y], color='blue', show=True)

if __name__ == '__main__':
    main()