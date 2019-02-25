import os
from PIL import Image

image_dir = '~/data/CropKidney256'
image_dir = os.path.expanduser(image_dir)
resize_image_size = (256, 256)


def main():
    for (path, dir, files) in os.walk(image_dir):
        for filename in files:
            if filename.endswith('.png'):
                image_file = os.path.join(path, filename)
                with open(image_file, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.resize(resize_image_size)
                        img.save(image_file)


if __name__ == '__main__':
    main()
