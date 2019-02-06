import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler as Scaler
import seaborn as sn
import time





df = pd.read_csv('diabetes.csv')
# print(df.describe())


print("The origin data set is like this: ")
print(df.head())
# print(df.shape)

# feature scaling
scaler = Scaler()
scaler.fit(df)
dataSetScaled = scaler.transform(df)
df = pd.DataFrame(data=dataSetScaled, columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
print("After feature scaling, the data set is like this: ")
print(df.head())



X = df.drop(columns=['Outcome'])


# print(X.head())
y = df['Outcome'].values
# print(y[0:5])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1, stratify =y)

start = time.time()
startCPU = time.clock()
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train, y_train)
# check accuracy of our model
print("accuracy:", knn.score(X_test, y_test))
end = time.time()
endCPU = time.clock()

print("Complete wall time:  " + str(end-start) + "Secs.")
print("Complete CPU time:  " + str(endCPU-startCPU) + "Secs.")

y_pred = knn.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pred)
print("confusion_matrix", cnf_matrix)

# plot the confusion matrix
df_cm = pd.DataFrame(cnf_matrix, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion matrix.png', dpi=400)




# # create a new KNN model for k-fold Cross-Validation
# knn_cv = KNeighborsClassifier(n_neighbors =3)
# cv_scores = cross_val_score(knn_cv, X, y, cv=5)
# print(cv_scores)
# print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# create a new knn model for GridSearchCV
knn2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X, y)
# check which n_neighbors value performed the best
print("best params:",knn_gscv.best_params_)
print("best accuracy:", knn_gscv.best_score_)

