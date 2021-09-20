import numpy as np 
import os
from skimage import feature as ft 
import cv2


def hogFeature(img_array, resize=(32,32)):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize)
    bins = 9
    cell_size = (8, 8)
    cpb = (2, 2)
    norm = "L2"
    features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size, 
                        cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
    return features

def extraHogFeaturesDir(img_dir, write_txt, per, type, resize=(32,32)):
    img_names = os.listdir(img_dir)
    print("Total Classes :",len(img_names))
    print("Getting all the Classes.....")
    noOfClasses=len(img_names)
    if os.path.exists(write_txt):
        os.remove(write_txt)
    count = 0
    with open(write_txt, "a") as f:
        index = 0
        for img in range (0,noOfClasses):
            myPicList = os.listdir(img_dir+"/"+str(count))
            countCheck = 0
            for img_name in myPicList:
                if type is "train":
                    if(countCheck < len(myPicList)*per):
                        img_array = cv2.imread(img_dir+"/"+str(count)+"/"+img_name)
                        features = hogFeature(img_array, resize)
                        label_num = count
                        row_data = img_name + "\t" + str(label_num) + "\t"
                        for element in features:
                            row_data = row_data + str(round(element,3)) + " "
                        row_data = row_data + "\n"
                        f.write(row_data)
                        countCheck += 1
                        if index%100 == 0:
                            print ("classes = ", len(img_names), "current image number = ", index)
                        index += 1
                    else:
                        break
                else:
                    if(countCheck >= len(myPicList)*(1 - per)):
                        img_array = cv2.imread(img_dir+"/"+str(count)+"/"+img_name)
                        features = hogFeature(img_array, resize)
                        label_num = count
                        row_data = img_name + "\t" + str(label_num) + "\t"
                        for element in features:
                            row_data = row_data + str(round(element,3)) + " "
                        row_data = row_data + "\n"
                        f.write(row_data)
                        countCheck += 1
                        if index%100 == 0:
                            print ("classes = ", len(img_names), "current image number = ", index)
                        index += 1
                    else:
                        countCheck += 1
            count += 1


if __name__ == "__main__":
    img_dir = "my_data_circles"
    writeTrain_txt = "hog/imgTrain_hog.txt"
    writeTest_txt = "hog/imgTest_hog.txt"
    extraHogFeaturesDir(img_dir, writeTrain_txt, 0.8, "train", resize=(32,32))
    extraHogFeaturesDir(img_dir, writeTest_txt, 0.2, "test", resize=(32,32))
   