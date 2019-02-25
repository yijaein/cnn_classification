from Tools.dicom_physical_size import convert_resize_physical, norm_path

# 원본 이미지 경로
src_path = norm_path('~/data/CropKidney')
# 저장할 이미지 경로
dst_path = norm_path('~/data/CropKidney_PhysicalSize(224px15cm)')
# 픽셀 사이즈가 들어있는 Dicom 정보 csv 파일
dicom_pixel_size_csv_path = '~/data/yonsei/doc/Dicom정보/dicom_info_100_400.csv'
# 리사이즈 크기
size_px = 224
# 리사이즈 크기에 대응할 실제 물리 크기
size_cm = 15

convert_resize_physical(src_path, dst_path, dicom_pixel_size_csv_path, size_px, size_cm)
