from KNN_distance import modified_hausdorff_distance
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import collections
import os
from matplotlib import pyplot as plt

class KNN:
    def __init__(self,neighbor_num,task = "classification"):
        self.k = neighbor_num
        # self.train_x = None
        # self.train_y = None
        self.task = task

    # def fit(self,train_x,train_y):
    #     self.train_x = train_x
    #     self.train_y = train_y

    def predict(self,train_x,train_y,test_x):
        # if not self.train_x:
        #     raise Exception("predicting without fitting... please call .fit on training data at first")
        self.predictions = []
        print("================= Fitting Begins =====================")
        for i in tqdm(test_x):
            results = []
            for idx,j in enumerate(train_x):
                results.append((modified_hausdorff_distance(i,j), idx))
            results = sorted(results)[:self.k]
            results = [train_y[result[1]] for result in results]
            prediction = None
            if self.task == "classification":
                threshold = 0
                counts = {}
                for result in results:
                    counts[result] = counts.get(result,0) + 1
                    if counts[result] > threshold:
                        threshold = counts[result]
                        prediction = result
            elif self.task == "regression":
                prediction = sum(results) / len(results)
            else:
                raise Exception("task cannot be recognized, please in dicate taks = classification/regression")
            self.predictions.append(prediction)
        print("================= Fitting Ends =====================")
        return self.predictions

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    data = pd.read_csv("leaf_dataset6.csv")
    # tmp = data[data["indexNumber"] == 2]
    # plt.plot(tmp.X,tmp.Y)
    # plt.show()
    prev_index = float("inf")
    coors = []
    labels = []
    index_numbers = []
    speciesNames = []
    test_index_numbers = []
    test_speciesNames = []
    # print(data.iloc[0])
    print("================= Loading Data Begins =====================")
    for i in tqdm(data.itertuples()):
        # print(i)
        if i.indexNumber != prev_index:
            prev_index = i.indexNumber
            coors.append([])
            index_numbers.append(i.indexNumber)
            speciesNames.append(i.speciesName)
            labels.append(i.speicesNumber)
        coors[-1].append((i.X,i.Y))
    print("================= Loading Data Ends =====================")
    threshold = 0.8
    counter = collections.Counter(labels)
    threshold_couns = collections.Counter()
    train_x,test_x = [],[]
    train_y,test_y = [],[]
    print("================= Splitting Data Begins =====================")
    for idx,label in tqdm(enumerate(labels)):
        threshold_couns[label] += 1
        if threshold_couns[label] > counter[label] * threshold:
            test_index_numbers.append(index_numbers[idx])
            test_speciesNames.append(speciesNames[idx])
            test_x.append(coors[idx])
            test_y.append(label)
        else:
            train_x.append(coors[idx])
            train_y.append(label)
    print("================= Splitting Data Ends =====================")
    for kkk in [3,5,7,9,11,13]:
        knn = KNN(kkk)
        pred = knn.predict(train_x,train_y,test_x)
        # print(({"indexNumber":index_numbers,"speciesName":speciesNames,"true_label":labels,"predict_label":pred}))
        output = pd.DataFrame({"indexNumber":test_index_numbers,"speciesName":test_speciesNames,"true_label":test_y,"predict_label":pred})
        output.to_csv("knn_prediction.csv",index=None)
        # print(pred)
        print("Accuracy of {} Nearest Neighbors is".format(knn.k), accuracy_score(pred,test_y))

class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, points, depth=0):
        if len(points) == 0:
            self.root = None
        else:
            k = len(points[0])  # Assuming all points have the same dimensionality
            axis = depth % k
            sorted_points = sorted(points, key=lambda x: x[axis])
            median = len(sorted_points) // 2

            self.root = Node(sorted_points[median])
            self.root.left = KDTree(sorted_points[:median], depth + 1)
            self.root.right = KDTree(sorted_points[median + 1:], depth + 1)

def inorder_traversal(node):    
    if node:
        inorder_traversal(node.left)
        print(node.point)
        inorder_traversal(node.right)