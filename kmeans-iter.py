import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    for K in range(1, 16):
        print(K)
        iterations = 200 # run as many times as desired and overlay or average results
        cluster_sums = np.zeros((iterations))

        for i in range(iterations):
            tempdata = []

            with open('data.csv', 'r') as f:
                reader = csv.reader(f)

                for row in reader:
                    floats = [float(s) for s in row]
                    tempdata.append(floats);

            f.close()
            n = len(tempdata)
            data = np.zeros((n, 2)) 
            clusters = np.zeros((K))
            old_labels = np.zeros((n))
            new_labels = np.zeros((n)) 
            
            for j in range(n):
                data[j][0] = tempdata[j][0]
                data[j][1] = tempdata[j][1]

            clusters = random_initialize(K, data)
            new_labels = update_labels(K, data, clusters)

            while any(old_labels != new_labels):
                old_labels = new_labels
                clusters = update_clusters(K, data, clusters, new_labels)
                new_labels = update_labels(K, data, clusters)

            # compute descriptive statistics
            for k in range(K):
                for l in range(n):
                    if new_labels[l] == k:
                        cluster_sums[i] += square_distance(data[l][0], data[l][1], \
                                clusters[k][0],clusters[k][1])

        print(K, np.min(cluster_sums))
        #plt.plot(K, np.min(cluster_sums))
        plt.scatter(K, np.min(cluster_sums), 20, 'black', 's')

    plt.show()

# pick K random data points to use as initial cluster centers
def random_initialize(K, data):
    n = np.size(data, axis = 0)
    clusters = np.zeros((K, 2))
    indices = np.zeros((K))
    repeat = True

    for i in range(K):
        while repeat == True:
            r = int(np.random.uniform(0, n - 1)) 

            if r not in indices:
                clusters[i][0] = data[r][0]
                clusters[i][1] = data[r][1]
                indices[i] = r
                repeat = False
            else:
                repeat = True

        repeat = True

    return clusters

# for each data point, find its nearest cluster
def update_labels(K, data, clusters):
    n = np.size(data, axis = 0)
    labels = np.zeros((n))

    for i in range(n):
        min_distance = 10000000

        for j in range(K):
            # using euclidean distance between two points,
            # though it probably doesn't matter
            distance = np.sqrt(square_distance(data[i][0], data[i][1], \
                    clusters[j][0],clusters[j][1]))
            
            if distance < min_distance:
                min_distance = distance
                min_index = j

        labels[i] = min_index

    return labels

# for each cluster, loop over all data points and find their average position
# update that cluster with that position
def update_clusters(K, data, clusters, labels):
    new_clusters = np.zeros((K, 2))
    sums = np.zeros(K)
    n = np.size(data, axis = 0)

    for i in range(K):
        for j in range(n):
            if labels[j] == i:
                new_clusters[i][0] += data[j][0]
                new_clusters[i][1] += data[j][1]
                sums[i] += 1
    
    new_clusters[:,0] /= sums
    new_clusters[:,1] /= sums
    return new_clusters

def square_distance(p1, p2, q1, q2):
    return np.power(p1 - q1, 2) + np.power(p2 - q2, 2)

main()
