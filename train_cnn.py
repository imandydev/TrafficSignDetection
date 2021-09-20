import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os


# Load dataset
def loadData(path):
    count = 0
    images = []
    classNo = []
    myList = os.listdir(path)
    print("Tổng số nhãn :",len(myList))
    noOfClasses=len(myList)
    print(".....")
    for x in range (0,noOfClasses):
        myPicList = os.listdir(path+"/"+str(count))
        for y in myPicList:
            curImg = cv2.imread(path+"/"+str(count)+"/"+y)
            images.append(curImg)
            classNo.append(count)
        print(count, end =" ")
        count +=1
    print(" ")
    images = np.array(images)
    classNo = np.array(classNo)
    return images, classNo, noOfClasses

# Xử lý ảnh
def grayScale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayScale(img)     # Chuyển thành ảnh nhị phân
    img = equalize(img)      # Cân bằng sáng
    img = img/255            # Chuẩn hóa pixel 0 -> 255 thành 0 -> 1
    return img

def processData(images, classNo, noOfClasses):
    # X_train = mảng hình ảnh dùng để train
    # y_train = classes id tưởng ứng với x train
    X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2, shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

    # Xử lý tất cả ảnh và trả về list
    X_train=np.array(list(map(preprocessing,X_train)))  
    X_validation=np.array(list(map(preprocessing,X_validation)))
    X_test=np.array(list(map(preprocessing,X_test)))

    # chuyển thành mảng 1 chiều
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
    X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
    
    # One-Hot Encode labels
    y_train = to_categorical(y_train,noOfClasses)
    y_validation = to_categorical(y_validation,noOfClasses)
    y_test = to_categorical(y_test,noOfClasses)
    return X_train, y_train, X_validation, y_validation, X_test, y_test

# xây dựng model
def myModel(noOfClasses):
    no_Of_Filters=60
    imageDimensions = (32, 32, 3)
    size_of_Filter1=(5, 5) # kích thước ma trận kernel 1   
    size_of_Filter2=(3,3) # kích thước ma trận kernel 2
    size_of_pool=(2,2) # kích thước ma trận pool
    no_Of_Nodes = 500   # số đầu ra của layer dense
    
    model= Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter1, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) 

    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(noOfClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def train_cnn(X_train, y_train, X_validation, y_validation, X_test, y_test, noOfClasses):
    model = myModel(noOfClasses)
    history = model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_validation, y_validation))
    score =model.evaluate(X_test,y_test,verbose=0)
    print('Test Score:',score[0])
    print('Test Accuracy:',score[1])
    model.save('model/cnn_model.h5')

if __name__ =="__main__":
    path = "my_data_circles"
    images, classNo, noOfClasses = loadData(path)
    X_train, y_train, X_validation, y_validation, X_test, y_test = processData(images, classNo, noOfClasses)
    train_cnn(X_train, y_train, X_validation, y_validation, X_test, y_test, noOfClasses)
   

    
   