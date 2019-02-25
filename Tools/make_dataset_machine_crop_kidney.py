import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--input_machine_crop_data', default='~/lib/robin_yonsei/result_us1_seg_v2/3_crop_by_seg', help='path to dataset')
parser.add_argument('--result_machine_crop_kidney_data', default='~/data/CropKidney_made_Machine_v2_re', help='path to dataset')

# fix seg_data path
parser.add_argument('--seg_data', default='~/data/SegKidney_v2', help='path to dataset')
args = parser.parse_args()


def norm_path(path):
    path = os.path.expanduser(path)
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.abspath(path)

    return path


def image_files(path):
    path = norm_path(path)

    images = list()
    for (file_path, dirs, files) in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext.lower() not in ['.jpg', '.png']:
                continue
            images.append(os.path.join(file_path, file))
    return images


def sub_paths(path1, path2):
    common_path = os.path.commonpath([path1, path2])

    post_path = path2[len(common_path):]
    sub_path = post_path.split(os.path.sep)[1:]

    return sub_path


def main(args):
    crop_files = image_files(args.input_machine_crop_data)
    print('len crop_files', len(crop_files))

    for file in crop_files:
        name = os.path.split(file)[1]
        seg_file = os.path.join(args.seg_data, name)
        isKidney = os.path.exists(seg_file)

        if not isKidney:
            continue

        # export sub dir
        sub_path = sub_paths(args.input_machine_crop_data, os.path.split(file)[0])

        src = file
        dst = os.path.join(args.result_machine_crop_kidney_data, *sub_path)

        if not os.path.exists(dst):
            os.makedirs(dst)

        shutil.copy(src, dst)


if __name__ == '__main__':
    args.input_machine_crop_data = norm_path(args.input_machine_crop_data)
    args.result_machine_crop_kidney_data = norm_path(args.result_machine_crop_kidney_data)
    args.seg_data = norm_path(args.seg_data)

    main(args)
