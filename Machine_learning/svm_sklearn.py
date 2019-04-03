from sklearn import svm, metrics
import pandas as pd

# load data(csv)
train = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/MeanPixel/train.csv'
val = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/MeanPixel/val.csv'

#change label string to Int and join together normal and AKI
def change_label_num(label):
    label.loc[label['disease'] == 'normal', 'disease'] = 0
    label.loc[label['disease'] == 'AKI', 'disease'] = 0
    label.loc[label['disease'] == 'CKD', 'disease'] = 1

def readData(trainCsv, valCsv, dataColumn=['kidney', 'liver'], labelColumn=['disease']):
    data_csv = pd.read_csv(trainCsv,
                           usecols=dataColumn)
    label_csv = pd.read_csv(trainCsv,
                            usecols=labelColumn)
    change_label_num(label_csv)

    val_data_csv = pd.read_csv(valCsv,
                               usecols=dataColumn)
    val_label_csv = pd.read_csv(valCsv,
                                usecols=labelColumn)
    change_label_num(val_label_csv)

    return data_csv, label_csv, val_data_csv, val_label_csv

X_train, y_train, X_test, y_test = readData(train, val)

# SVM
print('=' * 50 + 'SVM' + '=' * 50)
classifier = svm.SVC(C=1, gamma=0.1)
#gamma=0.001 -> 0.5294117647058824
#gamma=0.01 -> 0.47058823529411764
#gamma=0.1 -> 0.5588235294117647
#gamma=1 -> 0.5588235294117647

#fit to the trainin data
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))
print('ACC:', metrics.classification.accuracy_score(y_test, y_pred))
print('=' * 50 + 'SVM' + '=' * 50)

