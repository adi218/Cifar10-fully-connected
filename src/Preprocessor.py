import cv2
import numpy as np
import pickle
import os
import sklearn.model_selection as sk
class Preprocessor:
    X_train_arr = []
    labels_train_arr = []
    X_test = None
    labels_test = None

    def load_data(self):
        dict = {}
        for file in os.listdir("./../data/"):
            with open("./../data/"+file, 'rb') as fo:
                if("test" in file):
                    test = pickle.load(fo, encoding='latin1')
                    self.X_test = test['data']
                    self.labels_test = test['labels']
                else:
                    dict = pickle.load(fo, encoding='latin1')
                    self.X_train_arr.append(dict['data'])
                    self.labels_train_arr.append(dict['labels'])
        X_train = self.X_train_arr[0]
        labels_train = self.labels_train_arr[0]
        for batch in self.X_train_arr[1:]:
            X_train = np.append(X_train, batch, axis=0)
        for batch in self.labels_train_arr[1:]:
            labels_train = np.append(labels_train, batch)

        return X_train, labels_train, self.X_test, self.labels_test





# p = Preprocessor()
#
# p.load_data()