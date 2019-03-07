import csv
import os

from Tools.make_dataset_patient_info import get_patient_info, norm_path, filter_key

# [AccNo, Diagnosis]
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


def main(seg_path=None, csv_data=None, result_path=None, norm=None, mode='Eval'):
    global isangmi_list
    isangmi_accno_list = [AccNo for [AccNo, Diagnosis] in isangmi_list]

    seg_path = norm_path(seg_path)
    result_path = norm_path(result_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # make dirs, and open output file
    if mode == 'Eval':
        result_eval_csv = os.path.join(result_path, 'patient_kidney_eval.csv')
        feval = open(result_eval_csv, 'wt')
    else:
        result_train_path = os.path.join(result_path, 'train')
        result_train_csv = os.path.join(result_train_path, 'patient_kidney_train_by_isangmi.csv')
        result_val_path = os.path.join(result_path, 'val')
        result_val_csv = os.path.join(result_val_path, 'patient_kidney_train_by_isangmi.csv')

        for check_exists_path in [result_train_path, result_val_path]:
            if not os.path.exists(check_exists_path):
                os.makedirs(check_exists_path)

        ftrain = open(result_train_csv, 'wt')
        fval = open(result_val_csv, 'wt')

    # get patient info with kidney size
    kidney_info_list = get_patient_info(seg_path, csv_data, {})

    fieldnames = ['AccNo', 'Diagnosis', 'KidneyLongCm', 'KidneyShortCm', 'Age', 'Sex', 'Height', 'Weight']
    # fieldnames = ['AccNo', 'Diagnosis', 'KidneyLongCm', 'KidneyShortCm']

    if mode == 'Eval':
        writer_eval = csv.DictWriter(feval, fieldnames=fieldnames)
        writer_eval.writeheader()
    else:
        writer_train = csv.DictWriter(ftrain, fieldnames=fieldnames)
        writer_train.writeheader()
        writer_val = csv.DictWriter(fval, fieldnames=fieldnames)
        writer_val.writeheader()

    for kidney_info in kidney_info_list:

        # if mode == 'TrainVal':
        #     # only using 100, 400 data
        #     if kidney_info['Date'] not in ['180718', '180725']:
        #         continue

        if 0.0 in [kidney_info['KidneyLongCm'], kidney_info['KidneyShortCm']]:
            print('unknow pixel size, AccNo', kidney_info['AccNo'], kidney_info['Diagnosis'])
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

        if mode == 'Eval':
            writer_eval.writerow(kidney_info)
        else:
            if AccNo in isangmi_accno_list:
                writer_val.writerow(kidney_info)
            else:
                writer_train.writerow(kidney_info)

    if mode == 'Eval':
        feval.close()
    else:
        ftrain.close()
        fval.close()


if __name__ == '__main__':
    seg_path = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegKidney_v3'
    result_path = '/home/bong6/data/kidney_patient_H_info_add_200_for_svm'

    norm = dict()
    norm['diagnosis'] = True
    norm['kidney_size'] = False
    norm['sex'] = True
    norm['age'] = False
    norm['height'] = False
    norm['weight'] = False

    mode = 'TrainVal'  # 'TrainVal' or 'Eval'

    main(seg_path=seg_path, result_path=result_path, norm=norm, mode=mode)
