import csv

from Tools import csv_search
from Tools.dicom_physical_size import object_wh
from Tools.utils import *


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

            seg_file = os.path.split(seg_dict[name])[1]
            kidney_sizes.append([LongCM, ShortCM, seg_file])

        if len(kidney_sizes) == 0:
            # It hasn't kidney us image
            continue

        LongCM, ShortCM, SegFile = sorted(kidney_sizes, key=lambda x: x[0], reverse=True)[0]
        patient_info.update({'KidneyLongCm': LongCM, 'KidneyShortCm': ShortCM, 'SegFile': SegFile})
        patient_info_list.append(patient_info)

    return patient_info_list


def filter_key(dict_item, key_list):
    new_dict_item = dict()
    for key in dict_item.keys():
        if key in key_list:
            new_dict_item[key] = dict_item[key]
    return new_dict_item


def main(seg_path, csv_data, search, result_csv, norm):
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
            kidney_info = filter_key(kidney_info, fieldnames)

            if 0.0 in [kidney_info['KidneyLongCm'], kidney_info['KidneyShortCm']]:
                continue

            if '' in [kidney_info['Age'],
                      kidney_info['Sex'],
                      kidney_info['Height'],
                      kidney_info['Weight']]:
                continue

            # norm kidney 3~20cm into 0f~1f
            if norm['kidney_size']:
                for kidney_size_key in ['KidneyLongCm', 'KidneyShortCm']:
                    kidney_info[kidney_size_key] = (kidney_info[kidney_size_key] - 3) / 17

            if norm['sex']:
                if kidney_info['Sex'].lower() == 'm':
                    kidney_info['Sex'] = '0'  # Man
                else:
                    kidney_info['Sex'] = '1'  # Feman

            if norm['diagnosis']:
                if kidney_info['Diagnosis'].lower() == 'ckd':
                    kidney_info['Diagnosis'] = '1'  # ckd
                else:
                    kidney_info['Diagnosis'] = '0'  # aki, normal
            if norm['age']:
                kidney_info['Age'] = float(kidney_info['Age']) // 2

            if norm['height']:
                kidney_info['Height'] = float(kidney_info['Height']) // 2

            if norm['weight']:
                kidney_info['Weight'] = float(kidney_info['Weight']) // 2

            writer.writerow(kidney_info)


if __name__ == '__main__':
    diagnosis_csv_path = '~/data/yonsei2/doc/진단정보/diagnosis_info_400+100+1200.csv'
    dicom_csv_path = '~/data/yonsei2/doc/Dicom정보/dicom_info_100+400+1200.csv'
    patient_csv_path = '~/data/yonsei2/doc/환자 정보/patient_info_400+100+1200.csv'
    csv_data = csv_search.Csv_data(diagnosis_csv_path=diagnosis_csv_path, dicom_csv_path=dicom_csv_path,
                                   patient_csv_path=patient_csv_path)

    seg_path = '/home/bong07/data/yonsei2/machine/dataset/SegKidney_isangmi_file'
    result_csv = '/home/bong07/data/yonsei2/machine/dataset/kidney_patient_info/val/result.csv'

    norm = dict()
    norm['diagnosis'] = True
    norm['kidney_size'] = True
    norm['sex'] = True
    norm['age'] = False
    norm['height'] = False
    norm['weight'] = False

    main(seg_path, csv_data, {}, result_csv, norm)
