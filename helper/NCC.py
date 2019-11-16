# tslearn
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
# sklearn
from sklearn.metrics import accuracy_score

import numpy as np

def NearestCentroidClassification(X_train, X_test, y_train_n, y_test_n, dataset_name):
    '''

    :param X_train: if using DTAN, should already be aligned
    :param X_test: if using DTAN, should already be aligned
    :param y_train_n: numerical labels (not one-hot)
    :param y_test_n: numerical labels (not one-hot)
    :param dataset_name:
    :return: test set NCC accuracy
    '''

    # vars and placeholders
    input_shape = X_train.shape[1:]
    n_classes = len(np.unique(y_train_n))
    class_names = np.unique(y_train_n, axis=0)

    aligned_means = np.zeros((n_classes, input_shape[0], input_shape[1]))
    ncc_labels = []

    # Train set within class Euclidean mean
    for class_num in class_names:
        train_class_idx = y_train_n == class_num # get indices
        X_train_aligned_within_class = X_train[train_class_idx]
        aligned_means[int(class_num), :] = np.mean(X_train_aligned_within_class, axis=0)
        ncc_labels.append(class_num)

    ncc_labels = np.asarray(ncc_labels)

    # Nearest neighbor classification - using euclidean distance
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="euclidean")
    knn_clf.fit(aligned_means, ncc_labels)

    predicted_labels = knn_clf.predict(X_test)
    acc = accuracy_score(y_test_n, predicted_labels)

    print(f"{dataset_name} - NCC results: {acc}")