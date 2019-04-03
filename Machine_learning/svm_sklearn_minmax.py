from sklearn.svm import SVC
from sklearn import metrics
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
#minmax
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
print("특성별 최소 값:\n{}".format(X_train_scaled.min(axis=0)))
print("특성별 최대 값:\n{}".format(X_train_scaled.max(axis=0)))

X_test_scaled = (X_test -min_on_training)/ range_on_training

svc = SVC(C=1000)

svc.fit(X_train_scaled,y_train)

y_pred = svc.predict(X_test_scaled)


print("Classification report for classifier %s:\n%s\n"
      % (svc, metrics.classification_report(y_test, y_pred)))
print('ACC:', metrics.classification.accuracy_score(y_test, y_pred))
