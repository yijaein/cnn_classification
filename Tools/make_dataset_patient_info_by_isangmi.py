import csv
import os

from Tools import csv_search
from Tools.mask_dataset_patient_info import get_patient_info, norm_path, filter_key

isangmi_list = [
    ["1510201925", "aki"], ["1511278732", "aki"],
    ["1611224677", "aki"], ["1704042416", "aki"],
    ["1705109833", "aki"], ["1707271727", "aki"],

    ["1406115440", "ckd"], ["1504237440", "ckd"],
    ["1505140095", "ckd"], ["1509297909", "ckd"],
    ["1509313382", "ckd"], ["1510299374", "ckd"],
    ["1512012210", "ckd"], ["1601081840", "ckd"],
    ["1602014843", "ckd"], ["1603112332", "ckd"],
    ["1603374400", "ckd"], ["1604149457", "ckd"],
    ["1605119602", "ckd"], ["1605304657", "ckd"],
    ["1607110868", "ckd"], ["1609106232", "ckd"],
    ["1701014162", "ckd"], ["1701146758", "ckd"],
    ["1704026532", "ckd"], ["1704248494", "ckd"],
    ["1706254531", "ckd"], ["1709068883", "ckd"],
    ["1709143919", "ckd"],

    ["1401143131", "normal"], ["1604307774", "normal"],
    ["1604330063", "normal"], ["1605197062", "normal"],
    ["1608143415", "normal"], ["1609325227", "normal"],
    ["1702001450", "normal"], ["1702125801", "normal"],
    ["1706322448", "normal"], ["1707211552", "normal"],
    ["1707311002", "normal"], ["1708193790", "normal"],
    ["1712006741", "normal"], ["1712068741", "normal"],
    ["1712245042", "normal"], ["1712421090", "normal"]]


def main(seg_path, csv_data, result_path, norm):
    global isangmi_list
    isangmi_accno_list = [AccNo for [AccNo, Diagnosis] in isangmi_list]

    seg_path = norm_path(seg_path)
    result_path = norm_path(result_path)

    result_train_path = os.path.join(result_path, 'train')
    result_train_csv = os.path.join(result_train_path, 'patient_kidney_train_by_isangmi.csv')

    result_val_path = os.path.join(result_path, 'val')
    result_val_csv = os.path.join(result_val_path, 'patient_kidney_train_by_isangmi.csv')

    for check_exists_path in [result_train_path, result_val_path]:
        if not os.path.exists(check_exists_path):
            os.makedirs(check_exists_path)

    # get patient info with kidney size
    kidney_info_list = get_patient_info(seg_path, csv_data, {})

    with open(result_train_csv, 'wt') as ftrain, open(result_val_csv, 'wt') as fval:
        fieldnames = ['AccNo', 'Diagnosis', 'KidneyLongCm', 'KidneyShortCm', 'Age', 'Sex', 'Height', 'Weight']
        # fieldnames = ['AccNo', 'Diagnosis', 'KidneyLongCm', 'KidneyShortCm']

        writer_train = csv.DictWriter(ftrain, fieldnames=fieldnames)
        writer_train.writeheader()
        writer_val = csv.DictWriter(fval, fieldnames=fieldnames)
        writer_val.writeheader()

        for kidney_info in kidney_info_list:

            # only using 100, 400 data
            if kidney_info['Date'] not in ['180718', '180725']:
                continue

            if 0.0 in [kidney_info['KidneyLongCm'], kidney_info['KidneyShortCm']]:
                continue

            kidney_info = filter_key(kidney_info, fieldnames)

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

            AccNo = kidney_info['AccNo']

            if AccNo in isangmi_accno_list:
                writer_val.writerow(kidney_info)
            else:
                writer_train.writerow(kidney_info)


if __name__ == '__main__':
    diagnosis_csv_path = '~/data/yonsei/doc/진단정보/diagnosis_info_400+100+1200.csv'
    dicom_csv_path = '~/data/yonsei/doc/Dicom정보/dicom_info_100+400.csv'
    patient_csv_path = '~/data/yonsei/doc/환자 정보/patient_info_400+100+1200.csv'
    csv_data = csv_search.Csv_data(diagnosis_csv_path=diagnosis_csv_path, dicom_csv_path=dicom_csv_path,
                                   patient_csv_path=patient_csv_path)

    seg_path = '~/data/SegKidney'
    result_path = '~/data/kidney_more_info/isangmi_2'

    norm = dict()
    norm['diagnosis'] = True
    norm['kidney_size'] = True
    norm['sex'] = True
    norm['age'] = False
    norm['height'] = False
    norm['weight'] = False

    main(seg_path, csv_data, result_path, norm)
