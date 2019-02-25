import os
import numpy as np

import cv2

image_path = '~/data/masked_kidney'
output_path = '~/data/SegKidney/b'

image_path = os.path.expanduser(image_path)
output_path = os.path.expanduser(output_path)

if not os.path.isdir(output_path):
    os.makedirs(output_path)

def contour_filling(image):

    image = image.astype(np.uint8)

    _, contours, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    areaArray = []

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    # first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    for i in reversed(range(len(areaArray))):
        contour = sorteddata[i][1]
        if i<2:
            cv2.drawContours(image, [contour], 0, 255, -1)
        else :
            cv2.drawContours(image, [contour], 0, 0, -1)

    return np.array(image)

def main():
    for (path, dir, files) in os.walk(image_path):
        for filename in files:
            if filename.endswith('.png'):
                image = cv2.imread(os.path.join(path, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                image = contour_filling(image)

                cv2.imwrite(os.path.join(output_path, filename), image)


if __name__ == '__main__':
    main()