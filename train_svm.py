import numpy as np 
from sklearn.svm import SVC
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def loadHogData(hog_txt):
    labels = []
    hog_features = []
    with open(hog_txt, "r") as f:
        data = f.readlines()
        for row_data in data:
            row_data = row_data.rstrip()
            img_path, label, hog_str = row_data.split("\t")
            hog_feature = hog_str.split(" ")
            hog_feature = [float(hog) for hog in hog_feature]
            labels.append(label)
            hog_features.append(hog_feature)
    return np.array(labels), np.array(hog_features)



def svmTrain(hog_features, labels, save_path):
    
    clf = SVC(C=10, probability = True)
    clf.fit(hog_features, labels)
    joblib.dump(clf, save_path)
    print ("finished.")

def svmTest(svm_model, hog_feature, labels):
    clf = joblib.load(svm_model)
    accuracy = clf.score(hog_feature, labels)
    return accuracy


if __name__ =="__main__":
    hog_train_txt = "hog/imgTrain_hog.txt"
    hog_test_txt = "hog/imgTest_hog.txt"
    model_path = "model/svm_model.pkl"    
    labels, hog_train_features = loadHogData(hog_train_txt)
    svmTrain(hog_train_features, labels, model_path)
    
    labels, hog_test_features = loadHogData(hog_test_txt)
    test_accuracy = svmTest(model_path, hog_test_features, labels)
    print ("test accuracy = ", test_accuracy)

