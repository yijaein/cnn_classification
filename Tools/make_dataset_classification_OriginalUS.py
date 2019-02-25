import os
import shutil

from tqdm import tqdm

from Tools.make_dataset_classification_CropyKidneyShape import kidney_file_set


# 신장과 비신장 분류를 위한 원본영상 데이터셋 생성


def norm_path(path):
    path = os.path.normpath(path)
    path = os.path.normcase(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def file_dict(path):
    path = norm_path(path)

    d = dict()
    for (root, dirs, files) in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in ['.jpg', '.png']:
                continue
            d[name] = os.path.join(root, file)
    return d


def main(seg_path, isangmi_data_path, kidney_list_csv, result_path, train=True):
    seg_path = norm_path(seg_path)
    isangmi_data_path = norm_path(isangmi_data_path)
    result_path = norm_path(result_path)
    kidney_list_csv = norm_path(kidney_list_csv)

    seg_name = file_dict(seg_path)
    isangmi_dict = file_dict(isangmi_data_path)
    kidney_set = kidney_file_set(kidney_list_csv)

    for name in tqdm(isangmi_dict.keys()):
        # has kidney?
        if train:
            has_kidney = name in seg_name.keys()
        else:
            has_kidney = name in kidney_set


        src = isangmi_dict[name]
        dst_path = os.path.join(result_path, 'kidney' if has_kidney else 'non-kidney')
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        shutil.copy(src, dst_path)


if __name__ == '__main__':
    kidney_list_csv = norm_path('~/data/yonsei/doc/기기별_정제_데이터_영상/기기별 정제 영상 리스트(전체)_3차.csv')
    seg_path = norm_path('~/data/SegKidney')

    isangmi_data_path = norm_path('~/data/US_isangmi_folder/val')
    result_path = norm_path('~/data/KorNK/OriginalUS/val')
    main(seg_path, isangmi_data_path, kidney_list_csv, result_path, train=False)

    isangmi_data_path = norm_path('~/data/US_isangmi_folder/train')
    result_path = norm_path('~/data/KorNK/OriginalUS/train')
    main(seg_path, isangmi_data_path, kidney_list_csv, result_path, train=True)


