import numpy as np
import cv2

# Camera settings
frameWidth = 600  # Chiều rộng khung hình video
frameHeight = 600  # Chiều cao khung hình video
brightness = 180  # Độ sáng của video

RADIUS = 50  # Bán kính tối thiểu để phát hiện biển báo giao thông

MAIN_FRAME_TITLE = "Live Camera"  # Tiêu đề của cửa sổ hiển thị video

def runCamera():
    # Mở camera
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)  # Thiết lập chiều rộng của khung hình
    cap.set(4, frameHeight)  # Thiết lập chiều cao của khung hình
    cap.set(10, brightness)  # Thiết lập độ sáng của khung hình

    # Vòng lặp để đọc khung hình từ camera
    while True:
        # Đọc khung hình
        success, imgOrignal = cap.read()  # Đọc một khung hình từ camera
        if not success:  # Kiểm tra xem khung hình có được đọc thành công không
            break

        # Chuyển đổi khung hình sang ảnh xám
        gray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)
        # Áp dụng bộ lọc làm mờ trung vị để giảm nhiễu
        img = cv2.medianBlur(gray, 37)

        # Phát hiện các hình tròn trong khung hình
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=40)

        if circles is not None:  # Kiểm tra xem có hình tròn nào được phát hiện không
            circles = np.round(circles[0, :]).astype("int")  # Làm tròn và chuyển đổi tọa độ và bán kính thành số nguyên
            # Vẽ các hình tròn lên màn hình
            for (x, y, r) in circles:  # Lặp qua từng hình tròn
                if r >= RADIUS:  # Kiểm tra điều kiện bán kính
                    # Vẽ hình tròn bên ngoài với màu xanh lá cây và độ dày 2
                    cv2.circle(imgOrignal, (x, y), r, (0, 255, 0), 2)
                    # Vẽ một điểm trung tâm với màu đỏ và độ dày 3
                    cv2.circle(imgOrignal, (x, y), 2, (0, 0, 255), 3)

        # Hiển thị khung hình đã vẽ lên màn hình
        cv2.imshow(MAIN_FRAME_TITLE, imgOrignal)

        # Nhấn phím 'Esc' để thoát khỏi màn hình
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  # Giải phóng tài nguyên camera
    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ hiển thị

# Chạy hàm runCamera để bắt đầu chương trình
runCamera()
