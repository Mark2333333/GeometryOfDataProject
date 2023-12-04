from KNN_distance import modified_hausdorff_distance
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import collections
import os

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
    data = pd.read_csv("Contours_GastropodShells.csv")
    prev_index = float("inf")
    coors = []
    labels = []
    # print(data.iloc[0])
    print("================= Loading Data Begins =====================")
    for i in tqdm(data.itertuples()):
        # print(i)
        if i.indexNumber != prev_index:
            prev_index = i.indexNumber
            coors.append([])
            labels.append(i.genusNumber)
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
            test_x.append(coors[idx])
            test_y.append(label)
        else:
            train_x.append(coors[idx])
            train_y.append(label)
    print("================= Splitting Data Ends =====================")
    knn = KNN(9)
    pred = knn.predict(train_x,train_y,test_x)
    # print(pred)
    print("Accuracy of KNN is", accuracy_score(pred,test_y))
