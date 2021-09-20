import os
import numpy as np
import cv2
from skimage import feature as ft
import joblib
from PIL import Image

# Camera
frameWidth= 600     
frameHeight = 600
brightness = 180

threshold = 0.7    # Độ chính xác

RADIUS = 50 # bán kính nhỏ nhất để xét biển báo

font = cv2.FONT_HERSHEY_SIMPLEX

# tiêu cự với F = (P * D)/W
# P = 190 pixel
# W = 4 cm
# D = 20 cm
knowWidth = 5 #cm
FOCALLENGTH = 950

# tính khoảng cách từ camera đến biển báo
def distanceToCamera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth

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

# Rút trích đặc trưng
def hogExtraAndSvmClass(proposal, clf, resize=(32, 32)):
	img = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, resize)
	bins = 9
	cell_size = (8, 8)
	cpb = (2, 2)
	norm = "L2"
	features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size,
		cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
	features = np.reshape(features, (1, -1))
	cls_prop= clf.predict_proba(features)
	cls_prop = cls_prop[0]
	return cls_prop


def runCamera():
        cap = cv2.VideoCapture(0)
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
        cap.set(10, brightness)
        clf = joblib.load("model/svm_model.pkl")
        while (True):
            dis, img = cap.read()
            output = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bin = cv2.medianBlur(gray,37)
            circles = cv2.HoughCircles(img_bin, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=40)
            multiImg = []
            if circles is not None:
                addX1 = addY1 = 30
                # làm tròn số
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    if r >=  RADIUS:
                        # y tâm - bán kính - 30
                        yDrop = y-r-addY1
                        # x tâm - bán kính - 30
                        xDrop = x-r-addX1
                        # xét ngoại lệ
                        if yDrop < 0:
                            yDrop = 0
                        if xDrop < 0:
                            xDrop = 0
                        width = xDrop + (2*r + 2*addY1)
                        height = yDrop + (2*r + 2*addX1)
                        proposal = output[yDrop:height, xDrop:width]
                        imgTemp = Image.fromarray(proposal)
                        w, h = imgTemp.size

                        # khoảng cách từ biển báo đến camera làm tròn 1 chữ số thập phân
                        distanceToCam = round(distanceToCamera(knowWidth, FOCALLENGTH, w),1)

                        # rút trích đặt trưng và nhận diện
                        # cls_prop = [0.01 0.08 0.04 0.1  0.01 0.07 0.02 0.01 0.04 0.01 0.07 0.04 0.02 0.02 0.16 0.02 0.03 0.22 0.04]
                        cls_prop= hogExtraAndSvmClass(proposal, clf)
                        cls_prop = np.round(cls_prop, 2)
                        cls_num = np.argmax(cls_prop)
                        # nếu % chính xác > % chính xác mong muốn
                        if cls_prop[cls_num] > threshold:
                             # Vẽ ra vòng tròn màu xanh
                            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
                            traf = cv2.imread('images_root/' + str(cls_num) + '.png')
                            traf = cv2.resize(traf, (200,200))
                            multiImg.append(traf)
                            cv2.putText(img, str(getClassNameCircle(cls_num)), (x - 75, y-r-40), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(img,"D = " + str(distanceToCam) + " cm", (x - 75, y-r-10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                
                if not multiImg:
                    traf =  cv2.imread('images_root/none.png')
                    traf = cv2.resize(traf, (200,200))
                    multiImg.append(traf)

                # hiển thị biển báo nhận diện được
                verti = np.concatenate(multiImg, axis=0)
                cv2.imshow('Traffic sign', verti)

            cv2.imshow("Live Camera", img)
            # Exit = q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break