import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data_dir = "/Users/weizhaoli/Desktop/Machine learning/project/projectMnist/"
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print("train.shape: ", train.shape)
    X_train = train.values[0:train_row, 1:]
    Y_train = train.values[0:train_row, 0]

    Pred_test = pd.read_csv(data_dir + "test.csv").values
    print("pred_test.shape: ", Pred_test.shape)

    return X_train, Y_train, Pred_test


# train_row = 5000
train_row = 42000
Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)

print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_test.shape)
print(Origin_X_train)

# # display the image 
# row = 5
# print(Origin_y_train)
# plt.imshow(Origin_X_train[row].reshape((28, 28)))
# plt.savefig("row of 5 image")


# # display the overview of the data set
# classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# rows = 4

# print(classes)
# for y, cls in enumerate(classes):
#     idxs = np.nonzero([i == y for i in Origin_y_train])
#     idxs = np.random.choice(idxs[0], rows)
#     for i , idx in enumerate(idxs):
#         plt_idx = i * len(classes) + y + 1
#         plt.subplot(rows, len(classes), plt_idx)
#         plt.imshow(Origin_X_train[idx].reshape((28, 28)))
#         plt.axis("off")
#         if i == 0:
#             plt.title(cls)
# plt.savefig("data sample")


X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train, Origin_y_train,test_size = 0.2, random_state =0)

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)

ans_k = 0

k_range = range(1, 10)
scores = []

for k in k_range:
    print("k= " + str(k) + "begin ")
    start = time.time()
    startCPU = time.clock()
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_vali)
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    end = time.time()
    endCPU = time.clock()
    print(classification_report(y_vali, y_pred))
    print(confusion_matrix(y_vali, y_pred))

    print("Complete wall time:  " + str(end-start) + "Secs.")
    print("Complete CPU time:  " + str(endCPU-startCPU) + "Secs.")



print("scores: ",scores)
plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.savefig("the range of k & accuracy")

