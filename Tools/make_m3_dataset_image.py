'''
Ckd, Aki, Normal 분류를 위한 신장 이미지 데이터셋 생성
(척도 유지, 신장의 중앙 정렬, 건당 최대 폭의 신장 영상만 사용)
신장 정렬 데이터셋 만드는 코드

'''

import os

import cv2
import numpy as np

from Tools.crop_by_seg import calculate_angle, rotate_bound
from Tools.csv_search import Per_patient_fast
from Tools.dicom_physical_size import resize_physical_unit
from Tools.isangmi_val_list import isangmi_accno_list
from Tools.utils import image_dict, norm_path
from Tools.crop_by_seg import find_bounding_square


def paste(dst, src, dst_x=None, dst_y=None, dst_w=None, dst_h=None, src_x=None, src_y=None, src_w=None, src_h=None):
    dst_size = dst.shape[:2]
    src_size = src.shape[:2]

    if not dst_x:
        dst_x = 0

    if not dst_y:
        dst_y = 0

    if not src_x:
        src_x = 0

    if not src_y:
        src_y = 0

    if dst_w is not None and src_w is None:
        src_w = dst_w
    elif dst_w is None and src_w is not None:
        dst_w = src_w
    elif dst_w is None and src_w is None:
        min_w = min(dst_size[1] - dst_x, src_size[1] - src_x)
        src_w = dst_w = min_w
    else:
        pass

    if dst_h is not None and src_h is None:
        src_h = dst_h
    elif dst_h is None and src_h is not None:
        dst_h = src_h
    elif dst_h is None and src_h is None:
        min_h = min(dst_size[0] - dst_y, src_size[0] - src_y)
        src_h = dst_h = min_h
    else:
        pass

    # negative correction
    if dst_x < 0:
        neg = -dst_x

        dst_x = 0
        dst_w = dst_w - neg

        src_x += neg
        src_w = min(src_w + neg, dst_size[1])

    if dst_y < 0:
        neg = -dst_y

        dst_y = 0
        dst_h = dst_h - neg

        src_y += neg
        src_h = min(src_h + neg, dst_size[0])

    img = dst.copy()
    img[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] = src[src_y:src_y + src_h, src_x:src_x + src_w]
    return img


def main(us_path, mask_path, result_path,
         scale=None, view_scale=None, use_top_n_greatest_width=True, size=None, horizontally=True, crop=False, denoise=True):
    '''
    center boolean: 신장의 중앙 정렬 여부
    scale float: 영상 축적 고정
    only_greatest_width boolean: 신장 영상중 가장 큰것만 사용할지 여부
    resize (W, H): 최종 이미지 크기
    '''

    us_dict = image_dict(norm_path(us_path))
    mask_dict = image_dict(norm_path(mask_path))
    result_path = norm_path(result_path, True)

    for patient, dicoms in Per_patient_fast():

        patient_datas = list()
        for dicom in dicoms:
            Name = dicom['Name'][:-4]  # remove file ext
            Diagnosis = dicom['Diagnosis']
            AccNo = dicom['AccNo']
            PhysicalUnitsXDirection = dicom['PhysicalUnitsXDirection']
            PhysicalUnitsYDirection = dicom['PhysicalUnitsYDirection']
            PhysicalDeltaY = dicom['PhysicalDeltaY']  # cm per pixels
            PhysicalDeltaX = dicom['PhysicalDeltaX']  # 1 픽셀이 몇 cm 인지 나타냄

            # checking the kidney
            if Name not in mask_dict:
                # pass the non-kidney images
                continue

            # for debug
            # if Name != '1.2.840.113663.1500.1.295077244.3.8.20101014.92756.625':
            #     continue

            # checking physical unit(mm)
            if PhysicalUnitsXDirection != '3' or PhysicalUnitsYDirection != '3' \
                    and not PhysicalDeltaY and not PhysicalDeltaX:
                continue

            PhysicalDeltaY = float(PhysicalDeltaY)
            PhysicalDeltaX = float(PhysicalDeltaX)

            # read image
            us_img = cv2.imread(us_dict[Name], cv2.IMREAD_GRAYSCALE)
            print(us_dict[Name])

            mask_img = cv2.imread(mask_dict[Name], cv2.IMREAD_GRAYSCALE)


            # assert scale != view_scale, "use only scale or view_scale"

            if horizontally:
                angle = calculate_angle(mask_img)
                mask_img = rotate_bound(mask_img, angle)
                us_img = rotate_bound(us_img, angle)

            if scale:
                us_img = resize_physical_unit(us_img, (PhysicalDeltaX, PhysicalDeltaY), scale)
                mask_img = resize_physical_unit(mask_img, (PhysicalDeltaX, PhysicalDeltaY), scale)

            if view_scale:
                bbX, bbY, bbW, bbH = find_bounding_square(mask_img)
                fx = view_scale[0] / bbW
                fy = view_scale[1] / bbH
                us_img = cv2.resize(us_img, None, fx=fx, fy=fy)
                mask_img = cv2.resize(mask_img, None, fx=fx, fy=fy)

            if size:
                #numpy 좌표값을 얻어낸다 -> argwhere
                mask_pts = np.argwhere(mask_img == 255)
                #평균을 구한다
                mask_cx, mask_cy = int(np.mean(mask_pts[:, 1])), int(np.mean(mask_pts[:, 0]))
                #캔버스를 만들고 붙혀넣는다
                us_canvas = np.zeros(size, dtype=np.uint8)
                mask_canvas = np.zeros(size, dtype=np.uint8)

                # # for debug
                # us_canvas[:] = 100
                # mask_canvas[:] = 100

                canvas_h, canvas_w = size
                canvas_cx, canvas_cy = canvas_w // 2, canvas_h // 2
                dst_x, dst_y = canvas_cx - mask_cx, canvas_cy - mask_cy

                us_img = paste(dst=us_canvas, src=us_img, dst_x=dst_x, dst_y=dst_y)
                mask_img = paste(dst=mask_canvas, src=mask_img, dst_x=dst_x, dst_y=dst_y)

            if crop:
                mask = mask_img.astype(np.bool)
                us_img = us_img * mask

            if denoise:
                us_img = cv2.fastNlMeansDenoising(us_img, None, 10, 7, 21)

            data = dict()
            data['us'] = us_img
            data['mask'] = mask_img
            data['info'] = dicom
            patient_datas.append(data)

        # sort by kidney value
        patient_datas = sorted(patient_datas, key=lambda x: np.count_nonzero(x['mask'] == 255), reverse=True)

        # use one or all
        if use_top_n_greatest_width:
            cut_n = use_top_n_greatest_width
            patient_datas = patient_datas[:cut_n]

        accNo = patient['AccNo']
        diagnosis = patient['Diagnosis']
        train_val = 'val' if accNo in isangmi_accno_list else 'train'
        save_path = os.path.join(result_path, train_val, diagnosis, accNo)

        '''       if args.preprocess_denoise:
                np_out = np.asarray(out)
                np_out = cv2.fastNlMeansDenoising(np_out, None, 10, 7, 21)
                out = Image.fromarray(np.uint8(np_out))

            return out'''



        # save image
        for order, data in enumerate(patient_datas):
            us_img = data['us']
            name = data['info']['Name'][:-4]

            save_file = os.path.join(save_path, name + '.png')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print(save_file)
            cv2.imwrite(save_file, us_img)

        # save order of images
        if patient_datas:
            save_file = os.path.join(save_path, 'order.txt')
            with open(save_file, 'wt') as f:
                for order, data in enumerate(patient_datas):
                    name = data['info']['Name'][:-4]
                    f.write(name + '\n')


if __name__ == '__main__':
    us_path = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded'
    mask_path = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegKidney_v3'
    result_path = '~/data/US_Denoise'

    image_size_px = 512
    #비율
    image_size_cm = 18

    # ----------------------------------------------------------

    # 척도1 (cm per pixel)
    # 척도2과 같이 서로 같이 사용할 수 없음
    scale = None
    # scale = (image_size_cm / image_size_px, image_size_cm / image_size_px)  # cm per pixel

    # 척도2 (고정크기, fixed pixel size)
    # 척도1과 같이 서로 같이 사용할 수 없음
    view_scale = None
    # view_scale = [int(image_size_px * 0.85), int(image_size_px * 0.5)]
    # view_scale = [int(image_size_px * 1.0), int(image_size_px * 1.0)]

    # 출력 이미지 크기, 크기가 지정되면 신장을 중앙 정렬함
    size = None
    # size = (image_size_px, image_size_px)

    # 수평 정렬 (세그먼테이션 기준)
    # horizontally = True
    horizontally = False

    # 건당 가장 큰 신장 몇개를 사용할것인지
    # use_top_n_greatest_width = None
    use_top_n_greatest_width = 1

    # 마스크 모양에 따라 크롭할 것인지 여부
    # crop = True
    crop = False
    denoise = True

    main(us_path=us_path,
         mask_path=mask_path,
         result_path=result_path,
         use_top_n_greatest_width=use_top_n_greatest_width,
         scale=scale,
         view_scale=view_scale,
         size=size,
         horizontally=horizontally,
         denoise=denoise,
         crop=crop)
