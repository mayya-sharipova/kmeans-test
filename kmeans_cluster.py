import struct
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np


dims = 784
n_labels = 10


def cluster(normalize=False, n_clusters=10):
    vectors = []
    with open("train_vecs.vec", "rb") as in_f:
        while True:
            data = in_f.read(dims * 4)
            if not data:
                break
            vector = struct.unpack('<' + 'f' * dims, data) 
            vectors.append(vector)
            assert len(vector) == dims
    vectors = np.array(vectors)
    if normalize:
        vectors = vectors.astype(float) / 255.


    true_vectors_labels = []
    with open("train_labels.vec", "rb") as in_f:
        while True:
            data = in_f.read(4)
            if not data:
                break
            label = struct.unpack('<i', data)[0]
            true_vectors_labels.append(label)
    assert len(true_vectors_labels) == len(vectors)
	
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)
    vectors_labels = kmeans.labels_

    # Map clusters to labels
    labelCnts = np.zeros((n_clusters, n_labels), dtype=int)
    for i in range(len(vectors_labels)):
        cluster = vectors_labels[i]
        label = true_vectors_labels[i]
        labelCnts[cluster][label] += 1
    cluster_labels = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        maxCount = 0
        for j in range(n_labels):
            if labelCnts[i][j] > maxCount:
                maxCount = labelCnts[i][j]
                cluster_labels[i] = j
        #print(f"{i}: {labelCnts[i]}, max label: {cluster_labels[i]}")

    # Calculate accuracy
    transformed_labels = [cluster_labels[label] for label in vectors_labels]
    accuracy = accuracy_score(true_vectors_labels, transformed_labels)
    print(f"Accuracy for {n_clusters} clusters : {accuracy}")


n_clusters = [10, 16, 64, 256]
for value in n_clusters:
    cluster(False, n_clusters=value)
