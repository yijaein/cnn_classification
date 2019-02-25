import os
import random
from PIL import Image

data_dir = 'D:\\임무2_얼굴합성_데이터'
output_dir = 'D:\\fakeface4\\train\\1'
seg_dir = 'D:\\fakeface4_seg'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if not os.path.isdir(seg_dir):
    os.makedirs(seg_dir)

random_select = 14924 # image_per_label: 14924(31 classes), entire_normal_image: 462660

count = 1
for dir in os.listdir(data_dir):
    dir = os.path.join(data_dir, dir)
    if not os.path.isdir(dir):
        continue

    print (dir, 'reading...')

    image_files = []
    for filename in os.listdir(dir):
        ext = os.path.splitext(filename)[-1]
        if ext == '.png':
            image_files.append(os.path.join(dir, filename))

    # random select images
    random_select_img = random.sample(image_files, random_select)

    print (dir, 'processing...')

    # split image
    for image_path in random_select_img:
        img = Image.open(image_path)
        width, height = img.size[0], img.size[1]
        bbox = (0, 0, width // 2, height)
        slice_a_img = img.crop(bbox)

        bbox = (width // 2, 0, width, height)
        slice_b_img = img.crop(bbox)

        slice_a_img.save(os.path.join(output_dir, str(count) + '.png'))
        slice_b_img.save(os.path.join(seg_dir, str(count) + '.png'))
        count += 1

print ('count', count)