# Eetu Jyrkkänen

import random
from random import shuffle
from time import process_time
import timeit
import numpy as np
import matplotlib.pyplot as plt


def centroid(data, k):
    tmp_data = data.copy()
    shuffle(tmp_data)
    centroids = tmp_data[:k]

    with open('centroid.txt', 'w') as f:
        for centroid in centroids:
            tmp = [str(c) for c in centroid]
            f.write(' '.join(tmp))
            f.write('\n')
    return centroids


# remember to fix!
def euclideanDistance(obj1, obj2):
    return np.sqrt((obj1[0] - obj2[0]) ** 2 + (obj1[1] - obj2[1]) ** 2)


# Finds the index of nearest neighbor for data point from data space
def nearestNeighbor(dataPoint, dataSpace):
    index = 0
    dist = float('inf')

    for i in range(len(dataSpace)):
        tmp = euclideanDistance(dataPoint, dataSpace[i])
        if tmp < dist:
            dist = tmp
            index = i
    return index


# Assign data point(s) to its nearest centroid
def optimalPartition(data, centroids):
    partitions = []

    for dataPoint in data:
        partition = nearestNeighbor(dataPoint, centroids)
        partitions.append(partition + 1)

    with open('partition.txt', 'w') as f:
        for partition in partitions:
            f.write(str(partition))
            f.write('\n')

    return partitions


def localRepartition(data, centroids, partitions, oldVectors):
    tmp_partitions = partitions.copy()

    for dataPoint in oldVectors:
        partition = nearestNeighbor(data[dataPoint], centroids)
        tmp_partitions[dataPoint] = partition
    return tmp_partitions


# Builds new cluster from dimension averages
def buildCentroid(dataSpace):
    centroid = [0] * len(dataSpace[0])

    for i in range(len(dataSpace)):
        for j in range(len(dataSpace[i])):
            centroid[j] = centroid[j] + dataSpace[i][j] / len(dataSpace)

    return centroid


# returns SSE for each cluster
def SSE(data, centroids, partitions):
    tmp_data = [[] for i in range(len(centroids))]
    partition_sse = [0] * len(centroids)

    for i in range(len(partitions)):  # Gather data from each partition
        tmp_data[partitions[i] - 1].append(data[i])

    for i in range(len(centroids)):
        for j in range(len(tmp_data[i])):
            partition_sse[i] = partition_sse[i] + euclideanDistance(tmp_data[i][j], centroids[i]) ** 2

    return partition_sse


def kmeans(data, centroids, partitions, iterations=100):

    log = []  # .[0] sse, .[1] active centroids abs., .[2] active centroids %
    k = len(centroids)
    centroids = centroids
    partitions = partitions
    it = 0

    for iteration in range(iterations):

        it = iteration + 1
        sse = sum(SSE(data, centroids, partitions))
        act_centroids_abs = 0
        centroids_new = []
        tmp_data = [[] for i in range(k)]

        for i in range(len(partitions)):  # Gather data from each partition
            tmp_data[partitions[i] - 1].append(data[i])

        for i in range(k):  # Centroid step
            new_centroid = buildCentroid(tmp_data[i])
            centroids_new.append(new_centroid)

            if centroids[i] != new_centroid:  # Keeping track of active centroids
                act_centroids_abs = act_centroids_abs + 1

        centroids = centroids_new
        partitions = optimalPartition(data, centroids)

        # for logging
        sse_new = sum(SSE(data, centroids_new, partitions))
        log.append([sse_new, act_centroids_abs, act_centroids_abs / k * 100])
        #

        '''if iteration % 5 == 0:
            plot(data, centroids, sse_new, 'kmeans_' + str(iteration))'''
        if sse_new >= sse and act_centroids_abs == 0:
            #plot(data, centroids, sse_new, 'kmeans_' + str(iteration))
            break

    #kmeansLog(log)
    #plotSSE(log, 'kmeans_sse_iter_' + str(iteration))
    #plotActivity(log, 'kmeans_cent_iter_' + str(iteration))

    return centroids, partitions, it


# random swap algorithm that uses timer and returns iterations for task 1
def randomSwapTimer(data, centroids, partitions, timer, runKmeans=True, t=100):  # t = swaps

    start = process_time()
    k = len(centroids)
    centroids = centroids
    partitions = partitions
    improvements = []  # .[0] iteration, .[1] old sse, .[2] new sse
    it = 0

    for i in range(t):

        it = i+1
        new_centroids = centroids.copy()
        centroid_idx = random.randint(0, k - 1)
        new_centroids[centroid_idx] = data[random.randint(0, len(data) - 1)]  # Random Swap

        indices = [r for r in range(len(partitions)) if partitions[r] == centroid_idx + 1]
        new_partitions = localRepartition(data, new_centroids, partitions, indices)  # Re-allocate old vectors

        for j in range(len(data)):  # Create new cluster

            dist_old = euclideanDistance(data[j], new_centroids[new_partitions[j] - 1]) ** 2
            dist_new = euclideanDistance(data[j], new_centroids[centroid_idx]) ** 2
            if dist_new < dist_old:
                new_partitions[j] = centroid_idx + 1

        if runKmeans:
            new_centroids, new_partitions, iter_k = kmeans(data, new_centroids, new_partitions, 2)  # Run kmeans

        sse_old = SSE(data, centroids, partitions)
        sse_new = SSE(data, new_centroids, new_partitions)

        if sse_new < sse_old:
            centroids = new_centroids
            partitions = new_partitions
            improvements.append([i, sse_old, sse_new])
            #plot(data, centroids, sse_new, "RandomSwap_" + str(i))

        if process_time() - start > timer:
            break

    #randomSwapLog(improvements)
    return centroids, partitions, it


# normal random swap
def randomSwap(data, centroids, partitions, runKmeans=True, t=100):  # t = swaps

    k = len(centroids)
    centroids = centroids
    partitions = partitions
    improvements = []  # .[0] iteration, .[1] old sse, .[2] new sse
    sse = 0

    for i in range(t):

        new_centroids = centroids.copy()
        centroid_idx = random.randint(0, k - 1)
        new_centroids[centroid_idx] = data[random.randint(0, len(data) - 1)]  # Random Swap

        indices = [r for r in range(len(partitions)) if partitions[r] == centroid_idx + 1]
        new_partitions = localRepartition(data, new_centroids, partitions, indices)  # Re-allocate old vectors

        for j in range(len(data)):  # Create new cluster

            dist_old = euclideanDistance(data[j], new_centroids[new_partitions[j] - 1]) ** 2
            dist_new = euclideanDistance(data[j], new_centroids[centroid_idx]) ** 2
            if dist_new < dist_old:
                new_partitions[j] = centroid_idx + 1

        if runKmeans:
            new_centroids, new_partitions, iter_k = kmeans(data, new_centroids, new_partitions, 2)  # Run kmeans

        sse = sum(SSE(data, centroids, partitions))
        sse_new = sum(SSE(data, new_centroids, new_partitions))

        if sse_new < sse:
            centroids = new_centroids
            partitions = new_partitions
            improvements.append([i, sse, sse_new])
            sse = sse_new
            plot(data, centroids, sse_new, "RandomSwap_" + str(i))


    randomSwapLog(improvements)
    return centroids, partitions, sse


# random swap that utilizes SSE within clusters when selecting new centroid
def notSoRandomSwap(data, centroids, partitions, runKmeans=True, t=100):  # t = swaps

    k = len(centroids)
    centroids = centroids
    partitions = partitions
    improvements = []  # .[0] iteration, .[1] old sse, .[2] new sse
    sse = SSE(data, centroids, partitions)

    for i in range(t):

        new_centroids = centroids.copy()

        # find cluster with highest sse
        min = float('-inf')
        tmp_idx = 0
        for j in range(len(sse)):
            if sse[j] > min:
                min = sse[j]
                tmp_idx = j

        # gather indices of all the data points that belong to the winning cluster
        indices = [r for r in range(len(partitions)) if partitions[r] == tmp_idx + 1]
        random_idx = indices[random.randint(0, len(indices) - 1)]  # select new centroid
        centroid_idx = random.randint(0, k - 1)
        new_centroids[centroid_idx] = data[random_idx]  # Random Swap

        indices = [r for r in range(len(partitions)) if partitions[r] == centroid_idx + 1]
        new_partitions = localRepartition(data, new_centroids, partitions, indices)  # Re-allocate old vectors

        for j in range(len(data)):  # Create new cluster

            dist_old = euclideanDistance(data[j], new_centroids[new_partitions[j] - 1]) ** 2
            dist_new = euclideanDistance(data[j], new_centroids[centroid_idx]) ** 2
            if dist_new < dist_old:
                new_partitions[j] = centroid_idx + 1

        if runKmeans:
            new_centroids, new_partitions, iter_k = kmeans(data, new_centroids, new_partitions, 2)  # Run kmeans

        sse_new = SSE(data, new_centroids, new_partitions)

        if sum(sse_new) < sum(sse):

            centroids = new_centroids
            partitions = new_partitions
            improvements.append([i, sum(sse), sum(sse_new)])
            sse = sse_new
            plot(data, centroids, sum(sse_new), "notSoRandomSwap_" + str(i))


    randomSwapLog(improvements)
    return centroids, partitions, sum(sse)


def kmeansLog(log):
    print('---kmeans logs')
    for i in range(len(log)):
        print('iteration ', i, 'sse : ', log[i][0], 'active centroids : ', log[i][1], ' - ', log[i][2], '%')


def plot(data, centroids, sse, title):

    plt.title('SSE ' + str(sse))
    plt.scatter([i[0] for i in data], [i[1] for i in data], marker='o', c='red', s=15)
    plt.scatter([i[0] for i in centroids], [i[1] for i in centroids], marker='o',  c='blue', s=50)
    plt.savefig(title + '.png')
    plt.clf()


def plotSSE(logs, title):
    plt.title(title)
    plt.plot([i for i in range(len(logs))], [i[0] for i in logs])
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.savefig(title + '.png')
    plt.clf()


def plotActivity(logs, title):
    plt.title(title)
    plt.plot([i for i in range(len(logs))], [i[1] for i in logs])
    plt.xlabel("Iteration")
    plt.ylabel("Active centroids")
    plt.savefig(title + '.png')
    plt.clf()


def randomSwapLog(improvements):

    print('----RandomSwap log')
    for i in range(len(improvements)):
        print('Iteration ', improvements[i][0], ':', 'SSE before swap :', improvements[i][1], 'SSE after swap :',
              improvements[i][2])
    print('Total improvements : ', len(improvements))
    print('----')


# compares k-means and random swap
def timeTrial(data, k):

    repeats = 15
    log = [[] for i in range(repeats)]

    for i in range(repeats):
        centroids = centroid(data, k)
        partitions = optimalPartition(data, centroids)

        start = process_time()
        centroids_k, partitions_k, it_k = kmeans(data, centroids, partitions)
        elapsed = process_time() - start
        log[i].append(it_k)

        centroids_rs, partitions_rs, it_rs = randomSwapTimer(data, centroids, partitions, elapsed)
        log[i].append(it_rs)

    x = np.arange(len(log))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, [log[i][0] for i in range(len(log))], width, label='k-means')
    bar2 = ax.bar(x + width/2, [log[i][1] for i in range(len(log))], width, label='swaps')
    ax.set_ylabel('Iterations')
    ax.set_xlabel('Trial number')
    ax.set_title('Random swaps compared to k-means iterations')
    ax.set_xticks(x, x)

    ax.bar_label(bar1, padding=3)
    ax.bar_label(bar2, padding=3)
    fig.tight_layout()
    plt.savefig('timeTrial' + '.png')
    plt.clf()


def compareSwaps(data, k):
    repeats = 1
    log = [[] for i in range(repeats)]

    for i in range(repeats):

        centroids = centroid(data, k)
        partitions = optimalPartition(data, centroids)

        _centroids, _partitions, sse_rs = randomSwap(data, centroids, partitions, True, 20)
        log[i].append(sse_rs)
        _centroids, _partitions, sse_nsrw = notSoRandomSwap(data, centroids, partitions, True, 20)
        log[i].append(sse_nsrw)

    x = np.arange(len(log))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, [log[i][0] for i in range(len(log))], width, label='Random swap')
    bar2 = ax.bar(x + width/2, [log[i][1] for i in range(len(log))], width, label='Not so random swap')
    ax.set_ylabel('sse')
    ax.set_xlabel('Trial number')
    ax.set_title('Random swap compares to sse swap')
    ax.set_xticks(x, x)

    #ax.bar_label(bar1, padding=3)
    #ax.bar_label(bar2, padding=3)
    fig.tight_layout()
    plt.savefig('RandomSwapCompare' + '.png')
    plt.clf()


if __name__ == '__main__':

    data = []

    file = open('s1.txt', 'r')
    for line in file:
        data.append([float(f) for f in line.split()])

    file.close()

    k = 15

    # compare normal random swap and own implementation of random swap
    compareSwaps(data, k)
    # compare how many random swaps in same time that kmeans requires
    '''timeTrial(data, k)

    centroids = centroid(data, k)
    partitions = optimalPartition(data, centroids)

    # params; data, centroids, partitions, iterations default = 100
    centroids, partitions, it = kmeans(data, centroids, partitions)

    # params; (data, centroids, partitions, run kmeans default = True, swaps)
    randomSwap(data, centroids, partitions, False, 100)'''
