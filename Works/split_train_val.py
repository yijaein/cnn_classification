import os

us_path = ''
val_path = ''
des_path =''
image_files = list()
for (path, dir, files) in os.walk(us_path):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext != '.png' and ext != '.jpg':
            continue
        image_files.append(os.path.join(path, file))
