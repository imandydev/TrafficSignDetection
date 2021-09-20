import numpy as np
import cv2
import pickle
from keras.models import load_model
from PIL import Image

# camera
frameWidth= 600       
frameHeight = 600
brightness = 180

threshold = 0.9         # Độ chính xác

font = cv2.FONT_HERSHEY_SIMPLEX


RADIUS = 50 # bán kính nhỏ nhất để xét biển báo

# tiêu cự với F = (P * D)/W
# P = 190 pixel
# W = 4 cm
# D = 20 cm
knowWidth = 5 #cm
FOCALLENGTH = 950

# xử lý ảnh
def grayScale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayScale(img)
    img = equalize(img)
    img = img/255
    return img

   
def getClassNameCircle(classNo):
    if classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'Speed Limit 100 km/h'
    elif classNo == 7: return 'Speed Limit 120 km/h'
    elif classNo == 8: return 'No vehicles'
    elif classNo == 9: return 'No entry'
    elif classNo == 10: return 'Turn right ahead'
    elif classNo == 11: return 'Turn left ahead'
    elif classNo == 12: return 'Ahead only'
    elif classNo == 13: return 'Go straight or right'
    elif classNo == 14: return 'Go straight or left'
    elif classNo == 15: return 'Keep right'
    elif classNo == 16: return 'Keep left'
    elif classNo == 17: return 'Roundabout mandatory'
    else: return 'none'

# tính khoảng cách từ camera đến biển báo
def distanceToCamera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth

def runCamera():
    # Thiết lập camera
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)
    # Load file model
    model = load_model('model/cnn_model.h5')
    while True:
        # Đọc anh real time
        success, imgOrignal = cap.read()
        output = imgOrignal.copy()
        # Xử lý ảnh để tìm vòng tròn
        gray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(gray,37)
        # tìm vòng tròn
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                                1, 50, param1=120, param2=40)
        multiImg = []
        #  nếu tìm thấy vòng tròn
        if circles is not None:
            # danh sách biển báo [stt, tên, % chính xác]
            addX1 = addY1 = 30
            circles = np.round(circles[0, :]).astype("int")
            # duyệt qua tất cả vòng tròn tìm được x, y: tọa độ tâm vòng tròn r: bán kính
            for (x, y, r) in circles:
                # nếu bán kính lớn hơn bán kính nhỏ nhất được cho
                if r >=  RADIUS:
                    yDrop = y-r-addY1
                    xDrop = x-r-addX1
                    if yDrop < 0:
                        yDrop = 0
                    if xDrop < 0:
                        xDrop = 0
                    # chiều dài và chiều rộng muốn cắt ảnh
                    width = xDrop + (2*r + 2*addY1)
                    height = yDrop + (2*r + 2*addX1)
                    # Cắt ảnh dựa trên ảnh màu từ camera
                    circle_drop = output[yDrop:height, xDrop:width]
                    result = cv2.GaussianBlur(circle_drop,(5,5),0)
                    # chuyển từ cv2 sang image
                    image_1 = Image.fromarray(result)
                    w,h = image_1.size
                    # khoảng cách từ biển báo đến camera làm tròn 1 chữ số thập phân
                    distanceToCam = round(distanceToCamera(knowWidth, FOCALLENGTH, w),1)
                    # xử lý ảnh màu
                    img = np.asarray(image_1)
                    img = cv2.resize(img, (32, 32))
                    img = preprocessing(img)
                    img = img.reshape(1, 32, 32, 1)
                    # Nhận diện
                    predictions = model.predict(img)
                    classIndex = model.predict_classes(img)
                    probabilityValue =np.amax(predictions)
                    if probabilityValue > threshold:
                        # xóa []
                        classTraf = str(classIndex).replace("[","")
                        classTraf = str(classTraf).replace("]","")
                        trafImg = cv2.imread('images_root/' + str(classTraf) + '.png')
                        trafImg= cv2.resize(trafImg, (200,200))
                        multiImg.append(trafImg)
                        cv2.circle(imgOrignal, (x, y), r, (0, 255, 0), 2)
                        cv2.circle(imgOrignal, (x, y), 2, (0, 0, 255), 3)
                        cv2.putText(imgOrignal, str(getClassNameCircle(classIndex)), (x - 75, y-r-40), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(imgOrignal,"D = " + str(distanceToCam) + " cm", (x - 75, y-r-10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
           
            if not multiImg:
                trafImg =  cv2.imread('images_root/none.png')
                trafImg = cv2.resize(trafImg, (200,200))
                multiImg.append(trafImg)        
            # hiển thị biển báo nhận diện được
            verti = np.concatenate(multiImg, axis=0)
            cv2.imshow('Traffic sign', verti)

        cv2.imshow("Live Camera", imgOrignal)
        
        # Exit = q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

