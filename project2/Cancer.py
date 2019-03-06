import pandas as pd  
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import time



df = pd.read_csv('Cancer.csv')
# print(df.head())

df = df.drop("id", 1)
df = df.drop("Unnamed: 32", 1)
# print(df.head())

print(df.diagnosis.unique())
d = {'M':0,'B':1}
df['diagnosis'] = df['diagnosis'].map(d)
print(df.head())

# # draw the histogram 
# df.hist(bins=50, figsize=(20, 15))
# plt.savefig('hist.png')


# feature scaling
scaler = Scaler()
scaler.fit(df)
dataSetScaled = scaler.transform(df)

df = pd.DataFrame(data=dataSetScaled, 
columns = ['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 
'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])
print("After feature scaling, the data set is like this: ")
print(df.head())



# print(df.columns[1:31])

features = list(df.columns[1:31])
# print(features)

x = df[features]
y = df["diagnosis"]
# print(x)

# Split 
X_trainVal, X_test, y_trainVal, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)
# print(X_trainVal)
# print(y_trainVal)

# SVM classifier
svc = svm.SVC(kernel = 'linear', C=1).fit(X_trainVal,y_trainVal)


# K-fold cross-validation, testing the accuracy of the algorithm
kfold = KFold(n_splits = 5, shuffle = False)
print("KfoldCrossVal score using SVM is %s" %cross_val_score(svc, X_trainVal, y_trainVal, cv = kfold).mean())


# Testing
start = time.time()
startCPU = time.clock()

sm = svc.fit(X_trainVal, y_trainVal)
y_pred = sm.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pred)
print("confusion_matrix", cnf_matrix)

# plot the confusion matrix
# df_cm = pd.DataFrame(cnf_matrix, index = [i for i in "01"],
#                   columns = [i for i in "01"])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('confusion matrix.png', dpi=400)

print("accuracy_score is" , metrics.accuracy_score(y_test, y_pred))

end = time.time()
endCPU = time.clock()
print("Complete wall time:  " + str(end-start) + "Secs.")
print("Complete CPU time:  " + str(endCPU-startCPU) + "Secs.")