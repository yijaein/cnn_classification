import csv
import os
import shutil

from Tools.dicom_physical_size import norm_path, read_diagnosis_csv, read_dicom_pixel_size_csv, name_dict

'''
이상미 강사로 부터 컨펌 받은 신장 리스트를 기반으로 학습 데이터를 만드는 프로그램
컨펌 받은 신장 리스트는 각각의 파일이지만 해당 파일이 존재하는 폴더를 기반으로 함
단, US 전체 이미지를 사용하고(크롭 아님), 신장이 아닌 영상도 포함(폴더 내 모든 이미지 사용)
'''

# setting
src_path = norm_path('/media/bong07/895GB/data/yonsei/png_classification/once_400+100')
dst_path = norm_path('/home/bong07/data/yonsei/doc/기기별_정제_데이터_영상/US_isangmi_folder')
name_list_csv = norm_path('~/data/yonsei/doc/기기별_정제_데이터_영상/기기별 정제 영상 리스트(전체).csv')
diagnosis_csv = norm_path('~/data/yonsei/doc/진단정보/diagnosis_info_all(400+100+1200).csv')
dicom_info_csv = norm_path('~/data/yonsei/doc/Dicom정보/dicom_info_100_400.csv')


def read_name_list_csv(csv_file):
    l = list()
    with open(csv_file) as csvfile:
        csv_reader = csv.DictReader(csvfile)
        key_name = 'File'
        for line in csv_reader:
            name = line[key_name]
            l.append(name)

    return l


diagnosis_dict = read_diagnosis_csv(diagnosis_csv)
dicom_dict = read_dicom_pixel_size_csv(dicom_info_csv)
src_files_dict = name_dict(src_path)
val_name_list = read_name_list_csv(name_list_csv)

val_folder_list = set()
for name in val_name_list:
    folder = diagnosis_dict[name]['AccNo'].lower()
    val_folder_list.add(folder)

for name in src_files_dict:
    try:
        folder = diagnosis_dict[name]['AccNo'].lower()
        diagnosis = diagnosis_dict[name]['Diagnosis'].lower()
        manufacturer = dicom_dict[name]['Manufacturer'].lower()
    except KeyError:
        print('Not found key, ', name)
        continue

    if folder in val_folder_list:
        train_val = 'val'
    else:
        train_val = 'train'

    src = src_files_dict[name]
    dst = os.path.join(dst_path, train_val, diagnosis, manufacturer, folder)
    if not os.path.exists(dst):
        os.makedirs(dst)

    shutil.copy(src, dst)
    # print('copy', dst)
