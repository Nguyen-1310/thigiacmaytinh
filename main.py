import urllib.request as request
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def read_image_from_url(url):
    """Đọc ảnh từ URL và trả về dạng OpenCV"""
    try:
        req = request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error loading image:", e)
        return None

def add_gauss_noise(img, mean=0, sigma=20):
    """Thêm nhiễu Gaussian"""
    noise = np.random.normal(mean, sigma, img.shape)
    img_n = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img_n

def add_peper_noise(img, amount=0.02):
    """Thêm nhiễu muối tiêu"""
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)

    # white noise
    for _ in range(num_pixels):
        x, y = np.random.randint(0, h), np.random.randint(0, w)
        noisy[x, y] = 255

    # black noise
    for _ in range(num_pixels):
        x, y = np.random.randint(0, h), np.random.randint(0, w)
        noisy[x, y] = 0

    return noisy

def restore_img(img_noise):
    """Khử nhiễu bằng Gaussian blur"""
    _img = cv.GaussianBlur(img_noise, (3,3), 0)
    return _img

if __name__ == "__main__":
    # URL ảnh mẫu
    url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/lena.jpg"
    img = read_image_from_url(url)

    if img is None:
        exit()

    # Chuyển sang grayscale
    img_bw = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Phát hiện biên bằng Canny
    ed1 = cv.Canny(img, 100, 200)
    h, w = ed1.shape

    # Tạo mask ROI (tam giác phía dưới)
    mask = np.zeros_like(ed1)
    poly = np.array([[(0, h), (w, h), (w//2+50, h//2), (w//2 - 50, h//2)]], dtype=np.int32)
    cv.fillPoly(mask, poly, 255)
    roi = cv.bitwise_and(ed1, mask)

    # Hiển thị ROI bằng matplotlib
    plt.subplot(1,2,1); plt.imshow(ed1, cmap='gray'); plt.title("Canny")
    plt.subplot(1,2,2); plt.imshow(roi, cmap='gray'); plt.title("ROI")
    plt.show()

    # Hough Transform để phát hiện đường thẳng
    lines = cv.HoughLinesP(roi,
                           rho=1.0,
                           theta=np.pi/180,
                           threshold=50,
                           minLineLength=50,
                           maxLineGap=150)
    lane_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow("Detected Lanes", lane_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Thêm nhiễu muối tiêu
    img_pepper = add_peper_noise(img)
    cv.imshow("Pepper Noise", img_pepper)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Khử nhiễu bằng medianBlur (tốt cho muối tiêu)
    md_blur = cv.medianBlur(img_pepper, 5)
    cv.imshow("Median Blur", md_blur)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # So sánh edge detection trước và sau khử nhiễu
    edge_noisy = cv.Canny(img_pepper, 100, 200)
    edge_restored = cv.Canny(md_blur, 100, 200)
    cb_img = np.concatenate((edge_noisy, edge_restored), axis=1)
    cv.imshow("Edge Comparison", cb_img)
    cv.waitKey(0)
    cv.destroyAllWindows()