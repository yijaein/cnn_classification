from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics


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

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)

y_pred = forest.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (forest, metrics.classification_report(y_test, y_pred)))
print('ACC:', metrics.classification.accuracy_score(y_test, y_pred))