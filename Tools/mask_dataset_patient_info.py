import csv
import os
import cv2
from Tools.dicom_physical_size import read_diagnosis_csv, read_dicom_pixel_size_csv, object_wh
from Tools import csv_search

def norm_path(path):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


'''
지정한 이미지들의 정보를 출력

output: 폴더, 파일, 진단, 긴쪽, 짧은쪽

diagnosis_csv header: Date, File, RecoredPatientID, RealPatientID, AccNo, Diagnosis, Excluded
dicom_csv     header: File, Manufacturer, PhysicalUnitsXDirection, PhysicalDeltaX, PhysicalUnitsYDirection, PhysicalDeltaY
'''


def get_patient_info(seg_path, csv_data, search):

    # gather seg files
    seg_dict = dict()
    for (root, dirs, files) in os.walk(seg_path):
        for file in files:
            name_order, ext = os.path.splitext(file)

            if '#' in name_order:
                name, order = name_order.split('#')
                if order != '0':
                    continue
            else:
                name = name_order

            if ext.lower() not in ['.png', '.jpg']:
                continue
            seg_dict[name] = os.path.join(root, file)
    print('seg_dict', len(seg_dict))


    patient_info_list = list()
    for patient_info, dicom_info_list in csv_search.Per_patient(init_data=csv_data, search=search):
        kidney_sizes = list()
        for dicom_info in dicom_info_list:
            name = os.path.splitext(dicom_info['File'])[0]

            if name not in seg_dict:
                # It is non-kidney
                continue

            # compute Long cm, Short cm from kidney seg image
            seg_img = cv2.imread(seg_dict[name], cv2.IMREAD_GRAYSCALE)
            _, long_px, short_px = object_wh(seg_img)
            PhysicalDeltaX, PhysicalDeltaY = dicom_info['PhysicalDeltaX'], dicom_info['PhysicalDeltaY']
            PhysicalDeltaX = float(PhysicalDeltaX) if PhysicalDeltaX != '' else 0.0
            PhysicalDeltaY = float(PhysicalDeltaY) if PhysicalDeltaY != '' else 0.0
            LongCM, ShortCM = long_px * PhysicalDeltaX, short_px * PhysicalDeltaY

            kidney_sizes.append([LongCM, ShortCM, name])

        if len(kidney_sizes) == 0:
            # It hasn't kidney us image
            continue

        LongCM, ShortCM, File = sorted(kidney_sizes, key=lambda x: x[0], reverse=True)[0]
        patient_info.update({'KidneyLongCm': LongCM, 'KidneyShortCm': ShortCM, 'File': File})
        patient_info_list.append(patient_info)

    return patient_info_list


def filter_key(dict_item, key_list):
    new_dict_item = dict()
    for key in dict_item.keys():
        if key in key_list:
            new_dict_item[key] = dict_item[key]
    return new_dict_item


def main(seg_path, csv_data, search, result_csv):
    seg_path = norm_path(seg_path)
    result_csv = norm_path(result_csv)


    result_path = os.path.split(result_csv)[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # get patient info with kidney size
    kidney_info_list = get_patient_info(seg_path, csv_data, search)

    with open(result_csv, 'wt') as fcsv:
        fieldnames = ['AccNo', 'Diagnosis', 'KidneyLongCm', 'KidneyShortCm', 'Age', 'Sex', 'Height', 'Weight']
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for kidney_info in kidney_info_list:
            # print('1', kidney_info)
            kidney_info = filter_key(kidney_info, fieldnames)
            # print('2', kidney_info)
            writer.writerow(kidney_info)

if __name__ == '__main__':
    diagnosis_csv_path = '~/data/yonsei/doc/진단정보/diagnosis_info_400+100+1200.csv'
    dicom_csv_path = '~/data/yonsei/doc/Dicom정보/dicom_info_100+400.csv'
    patient_csv_path = '~/data/yonsei/doc/환자 정보/patient_info_400+100+1200.csv'
    csv_data = csv_search.Csv_data(diagnosis_csv_path=diagnosis_csv_path, dicom_csv_path=dicom_csv_path, patient_csv_path=patient_csv_path)

    seg_path = '~/data/SegKidney'
    result_csv = '~/data/kidney_more_info/train/patient_kidney_400.csv'
    #400건=180718, 100건=180725
    main(seg_path, csv_data, {'Date': '180718'}, result_csv)
