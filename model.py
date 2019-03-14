from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import os, os.path
import cv2

images = list()
target_class = list()

def load_images(directory):

    for folder in os.listdir(directory):

        if(folder == '.DS_Store' or folder == ".anonr" or folder == ".anon"):
            continue
        folder = os.path.join(directory, folder)

        for filename in os.listdir(folder):
            file = os.path.join(folder, filename)

            if(file.find("_4") !=  -1):
                if(file.find("sunglasses") != -1):
                    target_class.append(0)
                else:
                    target_class.append(1)
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    images.append(img.flatten())
    return images


load_images("/Users/anjithakarattuthodi/Desktop/faces")


X= np.array(images).reshape((len(images), -1))
y = target_class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33    )

mlp_model = MLPClassifier(hidden_layer_sizes=(25, 20, 20, 30), solver='lbfgs')
grid_search_model = GridSearchCV(mlp_model,{'learning_rate': ["constant", "invscaling", "adaptive"], 'activation': ["relu"]}, cv=5)
grid_search_model.fit(X_train, y_train)
y_pred = grid_search_model.predict(X_test)
target_names = ['sunglasses','open-eyes']

best_model=grid_search_model.best_estimator_
print("Training set score: %f" % grid_search_model.score(X_train, y_train))
print("Test set score: %f" % grid_search_model.score(X_test, y_test))
print("accuracy:\r\n",accuracy_score(y_pred,y_test))
print("CV_score train:\r\n", cross_val_score(best_model, X_train, y_train, cv=3))
print("CV_score test:\r\n", cross_val_score(best_model, X_test, y_test, cv=3))
print("Confusion Matrix:\r\n",confusion_matrix(y_test,y_pred))
print("Precision Recall Matrix:")
print(classification_report(y_test, y_pred,target_names=target_names))

y_score = best_model.predict_proba(X_test)
y_score = y_score[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_score.ravel())
plt.plot(recall, precision, marker='.')
print("Plotting Precision-Recall curve..")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
