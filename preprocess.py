import cv2
import numpy as np

def preprocess_image(image_path, output_path=None):
    # 1. 이미지 불러오기 & Grayscale 변환
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gamma 보정 (배경 어둡게, 뿌리 강조)
    gamma = 1.5
    gamma_corrected = np.power(gray / 255.0, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)

    # 3. 좌우 Crop (노이즈 제거용, 좌우 10% 제거)
    h, w = gamma_corrected.shape
    crop_ratio = 0.1
    cropped = gamma_corrected[:, int(w * crop_ratio):int(w * (1 - crop_ratio))]

    # 4. Gaussian Blur로 노이즈 제거
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)

    # 5. Otsu Threshold 적용 → 이진화
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. 가장 큰 윤곽선 찾아서 화분 위치 탐색
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pot_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(pot_contour)

    # 7. 화분 중심을 X축 중앙으로 이동 (중앙 정렬)
    center_x = cropped.shape[1] // 2
    object_center_x = x + w // 2
    shift_x = center_x - object_center_x
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    aligned = cv2.warpAffine(cropped, M, (cropped.shape[1], cropped.shape[0]))

    # 8. Morphological Opening으로 수직 뿌리 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 9. 뿌리 박스를 기준으로 아래쪽 자르기 (화분 제거)
    contours2, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours2:
        root_box = max(contours2, key=cv2.contourArea)
        _, y_root, _, h_root = cv2.boundingRect(root_box)
        final = aligned[:y_root + h_root, :]
    else:
        final = aligned  # fallback

    # 10. 추가 좌우 Crop (5%씩 더 제거)
    fh, fw = final.shape
    final_cropped = final[:, int(fw * 0.05):int(fw * 0.95)]

    # 저장 옵션
    if output_path:
        cv2.imwrite(output_path, final_cropped)

    return final_cropped
