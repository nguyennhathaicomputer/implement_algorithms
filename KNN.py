from sklearn import datasets
import numpy as np
import operator
import math

def calculate_accuracy(predicts, labels):
    correct_count = 0
    for i in range(len(predicts)):
        if predicts[i] == labels[i]:
            correct_count+=1
    accuracy = correct_count/len(predicts)
    return accuracy

def calculate_distance(p1, p2):
    dis = 0
    for i in range(len(p1)):
        dis+= (p1[i]-p2[i])*(p1[i]-p2[i])
    return math.sqrt(dis)
        


def get_k_neighbor(training_X, label_y, point, k):
    distances = []
    neighbors = []

    #calculate the distance from  point to another point in the training set
    for i in range(len(training_X)):
        dis = calculate_distance(point, training_X[i])
        distances.append((dis, label_y[i]))

    # sort tuple(distance, label_y) by distance
    distances.sort(key = operator.itemgetter(0))

    # get k label have distance closet
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def highest_vote(labels):
    count_label=[0,0,0]
    for label in labels:
        count_label[label]+=1
    max_label = max(count_label)
    best_label = count_label.index(max_label)
    return best_label

def predict(training_X, label_y, point, k):
    neighbors = get_k_neighbor(training_X, label_y, point,k)
    return highest_vote(neighbors)


iris = datasets.load_iris()
iris_X = iris.data
iris_y =  iris.target


# random and shuffle the data
randindex = np.arange(iris_X.shape[0])

np.random.shuffle(randindex)

iris_X = iris_X[randindex]
iris_y = iris_y[randindex]

# splitting the data into the training set and test set
X_train = iris_X[:100,:]
X_test = iris_X[100:,:]
y_train = iris_y[:100]
y_test = iris_y[100:]

y_pred = []
k = 5
for p in X_test:
    label = predict(X_train, y_train, p, k)
    y_pred.append(label)
print(y_pred)
print(y_test)

acc = calculate_accuracy(y_pred, y_test)
print(acc)