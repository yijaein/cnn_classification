import os
import argparse
import shutil
#made by JAN
seg_image_path = dict()
seg_image =[]
no_seg_image =[]

parser = argparse.ArgumentParser(description="pytorch kidney dataset make")
parser.add_argument('--data', default='/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/pngs/180718_KidneyUS_400_png')
parser.add_argument('--segimg_dir', default='/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegKidney_v3')
parser.add_argument('--data_dir', default='/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/pngs/kidney12')
args = parser.parse_args()

black_image_path = 'black.png'

def main():
    for (path, dir, files) in os.walk(args.segimg_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png' or ext == '.jpg':
                seg_image_path[filename] = os.path.join(path, filename)

    for (path, dir, files) in os.walk(args.data):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png' or ext == '.jpg':
                image_path = os.path.join(path, filename)
                if filename in seg_image_path:
                    seg_image.append((image_path, 0, seg_image_path[filename]))
                    if not os.path.exists(args.data_dir):
                        os.makedirs(args.data_dir)

                    shutil.copy(image_path,args.data_dir)
                    print(image_path)
                else:
                    no_seg_image.append((image_path, 1, black_image_path))
    # set samples


if __name__ == '__main__':
    main()

