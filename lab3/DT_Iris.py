import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def entropy(class_probabilities): # list of probabilities as input
    return sum(-p * math.log(p, 2) for p in class_probabilities if p > 0)

def data_entropy(labels):
    return entropy(class_probabilities(labels))

def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

class DataSet:
    def __init__(self, X=np.array([[]]), Y=np.array([])):
        self.X_data = X
        self.Y_data = Y
        self.feature_names = None

    def view_dataset_info(self):
        print("\n *** DataSet Start ***")
        for xd, yd in zip(self.X_data, self.Y_data):
            print(" Data : ", xd, " Class: ", yd)
        print("*** DataSet End ***\n")

class Node:
    def __init__(self, indx = -1, value = None):
        self.feature_indx = indx
        self.feature_name = ""
        self.node_value = value
        self.right_child = None
        self.left_child = None

    def view_node_info(self):
        print("\n *** Node Start ***")
        print("feature column : ", self.feature_name)
        print("feature indx : ", self.feature_indx)
        print("feature value : ", self.node_value)
        print("*** Node End ***\n")




class DTree:
    def __init__(self):
        self.root = Node()
        #self.temp = Node()
        self.train_data = None
        self.class_data = None
        self.predicted = np.array([])
        self.data_names = np.array([])

    def split_by(self, f_indx, fval,  DSet):
        npoints, nfeatures = DSet.X_data.shape

        DSet_left = DataSet()
        DSet_left.feature_names = DSet.feature_names.delete(f_indx)
        DSet_right = DataSet()
        DSet_right.feature_names = DSet.feature_names.delete(f_indx)

        for fentry, class_type in zip(DSet.X_data, DSet.Y_data):
            if type(fentry[f_indx]) != str:
                with_f_deleted = np.delete(fentry, f_indx)
                if fentry[f_indx] < fval:
                    DSet_left.Y_data = np.append(DSet_left.Y_data, class_type)
                    DSet_left.X_data = np.append(DSet_left.X_data, with_f_deleted)
                else:
                    DSet_right.X_data = np.append(DSet_right.X_data, with_f_deleted)
                    DSet_right.Y_data = np.append(DSet_right.Y_data, class_type)

        DSet_right.X_data = DSet_right.X_data.reshape(len(DSet_right.Y_data), nfeatures-1)
        DSet_left.X_data = DSet_left.X_data.reshape(len(DSet_left.Y_data), nfeatures - 1)

        return DSet_left, DSet_right

    def decide(self, DSet, decision_node):
        npoints, nfeatures = DSet.X_data.shape # npoints = num of entries in X_data,
        #DSet.view_dataset_info()            # nfeatures = num of features in X_data
        #input()

        if np.any(DSet.X_data) == False: # DSet.X_data is empty
            label_counts = Counter(DSet.Y_data)
            most_common_label = label_counts.most_common(1)[0][0]
            decision_node.feature_indx = -1
            decision_node.node_value = most_common_label
            print("leaf with class type : ", most_common_label)
            return

        if len(np.unique(DSet.Y_data)) == 1: # if only one class in dataset
            decision_node.feature_indx = -1
            decision_node.node_value = DSet.Y_data[0]
            print("leaf with class type : ", DSet.Y_data[0])
            return
        

        min_entropy = sys.maxsize  # min entropy of splitting

        DSetLeft = DataSet()
        DSetRight = DataSet()

        for findx in range(0, nfeatures):  # итеррируем по всем признакам
            x = DSet.X_data[:, findx]  # select current feature column
            x_unique_values = np.unique(x)  # remove duplicates from current column of features

            for current_feature_value in x_unique_values:  # смотрим разбиение по каждому уникальному признаку
                data_set_left, data_set_right = self.split_by(findx, current_feature_value, DSet) # разбиваем по каждому уникальному признаку
                entropy_of_partition = partition_entropy([data_set_left.Y_data, data_set_right.Y_data]) # считаем энтропию разбиения
                if min_entropy > entropy_of_partition:
                    min_entropy = entropy_of_partition
                    decision_node.node_value = current_feature_value
                    decision_node.feature_name = DSet.feature_names[findx]
                    decision_node.feature_indx = np.where(self.data_names == decision_node.feature_name)[0][0]
                    DSetLeft = data_set_left
                    DSetRight = data_set_right

        #print(" \n Splitting node info : ")
        #print(" Entropy of a split : ", min_entropy)
        decision_node.view_node_info()
        #input()
        decision_node.left_child = Node()
        decision_node.right_child = Node()
        self.decide(DSetLeft, decision_node.left_child)
        self.decide(DSetRight, decision_node.right_child)


    def traverse(self, entry, node, columns):
        if node.left_child == None and node.right_child == None:
            print("Class of : ", entry, " is -> ", node.node_value)
            self.predicted = np.append(self.predicted, node.node_value)
            return
        
        if type(node.node_value) != str:
            indx = columns.index(node.feature_name)
            if entry[indx] < node.node_value:
                self.traverse(entry, node.left_child, columns)
            else:
                self.traverse(entry, node.right_child, columns)
                
           
        
        
    def classify(self, data, answers=None):
        columns = data.columns.to_list()
        X_data = data.to_numpy()
        
        self.predicted = np.array([])
        for entry in X_data:
            indx = columns.index(self.root.feature_name)

            if type(self.root.node_value) != str:
                if entry[indx] < self.root.node_value:
                    self.traverse(entry, self.root.left_child, columns)
                else:
                    self.traverse(entry, self.root.right_child, columns)

        answers_data = np.array([])
        if type(answers) != None:
            answers_data = answers.to_numpy()
            le = LabelEncoder()
            answers_data = le.fit_transform(answers_data)
            eff = 0
            for p, r in zip(self.predicted, answers_data):
                print(" predicted : ", p, " real : ", r)
                if p == r:
                    eff+=1

            print(" efficiency : ", eff/len(self.predicted))
            cf_matrix = confusion_matrix(answers_data, self.predicted)
            sns.heatmap(cf_matrix, annot=True)
            plt.show()
                    
            
    def build_tree(self, train_data, class_data):
        self.root = Node()
        self.data_names = train_data.columns.to_numpy()
        self.train_data = train_data
        self.class_data = class_data

        X_data = train_data.to_numpy()
        Y_data = class_data.to_numpy()
        
        le = LabelEncoder()
        Y_data = le.fit_transform(Y_data)

        MainDataSet = DataSet(X_data, Y_data)
        MainDataSet.feature_names = X_train.columns

        self.decide(MainDataSet, self.root)


dt = pd.read_csv("Iris.csv")
#sns.pairplot(data=dt, hue = 'Species')
#plt.show()

X,Y=dt.drop(["Id","Species","PetalWidthCm"],axis=1),dt["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,  random_state=54)

Tree = DTree()
Tree.build_tree(X_train,  y_train)

#Tree.classify(X_train, y_train)
Tree.classify(X_test, y_test)




