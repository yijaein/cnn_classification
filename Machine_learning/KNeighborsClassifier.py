from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import mglearn
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

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, y_pred)))
print('ACC:', metrics.classification.accuracy_score(y_test, y_pred))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(train,val)
    mglearn.plots.plot_2d_separator(clf, train, fill=True, eps=0.5, ax=ax,
                                    alpha=.4)
    mglearn.discrete_scatter(train[:,0], train[:,1], val, ax=ax)
    ax.set_title("{}이웃".format(n_neighbors))
    ax.set_xlabel("feature1")
    ax.set_ylabel("feature2")
axes[0].legend(loc=3)



