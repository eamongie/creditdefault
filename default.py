import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
dataset = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
dataset.head()

dataset = dataset.rename(columns={'PAY_0':'PAY_1', 'default.payment.next.month':'DEFAULT'})
dataset.head()

for col in dataset:
    print(col, dataset[col].unique())
    
others = (dataset.EDUCATION == 5) | (dataset.EDUCATION == 6) | (dataset.EDUCATION == 0)
dataset.loc[others, 'EDUCATION'] = 4
print('EDUCATION', dataset.EDUCATION.unique())

dataset.loc[dataset.MARRIAGE == 0, 'MARRIAGE'] = 3
print('MARRIAGE', dataset.MARRIAGE.unique())

others = (dataset.PAY_1 == 0) | (dataset.PAY_1 == -1) | (dataset.PAY_1 == -2)
dataset.loc[others, 'PAY_1'] = 0
print('PAY_1', dataset.PAY_1.unique())

others = (dataset.PAY_2 == 0) | (dataset.PAY_2 == -1) | (dataset.PAY_2 == -2)
dataset.loc[others, 'PAY_2'] = 0
print('PAY_2', dataset.PAY_2.unique())

others = (dataset.PAY_3 == 0) | (dataset.PAY_3 == -1) | (dataset.PAY_3 == -2)
dataset.loc[others, 'PAY_3'] = 0
print('PAY_3', dataset.PAY_3.unique())

others = (dataset.PAY_4 == 0) | (dataset.PAY_4 == -1) | (dataset.PAY_4 == -2)
dataset.loc[others, 'PAY_4'] = 0
print('PAY_4', dataset.PAY_4.unique())

others = (dataset.PAY_5 == 0) | (dataset.PAY_5 == -1) | (dataset.PAY_5 == -2)
dataset.loc[others, 'PAY_5'] = 0
print('PAY_5', dataset.PAY_5.unique())

others = (dataset.PAY_6 == 0) | (dataset.PAY_6 == -1) | (dataset.PAY_6 == -2)
dataset.loc[others, 'PAY_6'] = 0
print('PAY_6', dataset.PAY_6.unique())

SUM_BILLS = dataset["BILL_AMT1"] + dataset["BILL_AMT2"] + dataset["BILL_AMT3"] + dataset["BILL_AMT4"] + dataset["BILL_AMT5"] + dataset["BILL_AMT6"]
SUM_PAYS = dataset["PAY_AMT1"] + dataset["PAY_AMT2"] + dataset["PAY_AMT3"] + dataset["PAY_AMT4"] + dataset["PAY_AMT5"] + dataset["PAY_AMT6"]
BILL_PAY_DIFF = SUM_BILLS - SUM_PAYS
dataset["BILL_PAY_DIFF"] = BILL_PAY_DIFF
CREDIT_UTIL = dataset["BILL_PAY_DIFF"] / dataset["LIMIT_BAL"]
dataset["CREDIT_UTIL"] = CREDIT_UTIL
# dataset = dataset.drop(columns=["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"])
# dataset = dataset.drop(columns=["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"])
dataset = dataset.drop(columns=["ID"])
print(dataset)

corr = dataset.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

card = dataset
x, y = card.drop('DEFAULT',axis=1), card.DEFAULT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

#create an array of models
models = []
models.append(("Logistic Regression",LogisticRegression(max_iter=100000)))
models.append(("Naive Bayes",GaussianNB()))
models.append(("Random Forest",RandomForestClassifier()))
models.append(("Support Vector",SVC()))
models.append(("Decision Tree",DecisionTreeClassifier()))
models.append(("KNN",KNeighborsClassifier()))

#measure the accuracy 
for name,model in models:
    kfold = KFold(n_splits=2)
    cv_result = cross_val_score(model,x_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result)
    
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train,y_train)
print('train rf.score: ', rf.score(x_train,y_train))
print('test rf.score: ', rf.score(x_test,y_test))
final_pred = rf.predict(x_test)

from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import auc

false_positive, true_positive, threshold = roc_curve(y_test, final_pred)
roc_auc = auc(false_positive, true_positive)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive, true_positive, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of RF')
plt.show()

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

fmeasure1 = f1_score(y_test, final_pred, average="macro")
fmeasure2 = f1_score(y_test, final_pred, average="micro")

precision = precision_score(y_test, final_pred, average="macro")
recall = recall_score(y_test, final_pred, average="macro")
f1 = 2*(precision*recall)/(precision + recall)

print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()
print(classification_report(y_test, final_pred))

bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5, max_features=1.0, n_estimators=20)
bg.fit(x_train,y_train)
bg.score(x_train,y_train)
bg.score(x_test,y_test)
print('train bagging score: ', bg.score(x_train,y_train))
print('test bagging score: ', bg.score(x_test,y_test))
final_pred = bg.predict(x_test)

from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import auc

false_positive, true_positive, threshold = roc_curve(y_test, final_pred)
roc_auc = auc(false_positive, true_positive)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive, true_positive, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of Bagging')
plt.show()

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

fmeasure1 = f1_score(y_test, final_pred, average="macro")
fmeasure2 = f1_score(y_test, final_pred, average="micro")

precision = precision_score(y_test, final_pred, average="macro")
recall = recall_score(y_test, final_pred, average="macro")
f1 = 2*(precision*recall)/(precision + recall)

print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()
print(classification_report(y_test, final_pred))

ad = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=0.01)
ad.fit(x_train,y_train)
ad.score(x_train,y_train)
ad.score(x_test,y_test)
print('train adaboost score: ', ad.score(x_train,y_train))
print('test adaboost score: ', ad.score(x_test,y_test))

