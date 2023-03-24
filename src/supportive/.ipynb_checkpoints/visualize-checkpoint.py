# Author: Lennart Seeger
# This file is licensed under the Apache 2.0 license available at
# http://www.apache.org/licenses/LICENSE-2.0
import numpy as np
import sys
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt

sys.path.insert(1, '../src')

"""
    Function calculating feature vectors for a method and a dataset

    Attributes
    ----------
    method: method
        method that should be used for feature extraction
    x_test: array
        test dataset
    embedding_size: number
        size of the feature vectors
    image_size: number
        size of the images
    
    Returns
    ----------
    x_test_vectors: array
        feature vectors of input images
"""
def predict_vectors(method, x_test, embedding_size=2048, image_size=256):
    counter = 0
    x_test_vectors = np.empty([1, embedding_size])
    for element in x_test:
        counter += 1
        if counter % 100 == 0:
            print(counter)
        x_test_vectors = np.append(x_test_vectors,
                                   method(element.reshape(1, image_size, image_size, 3)),
                                   axis=0)
    x_test_vectors = x_test_vectors[1:]
    return (x_test_vectors)


"""def visualize_predictions(x_test_vectors, x_test):
    from scipy.spatial.distance import cosine
    # automated multiple executions of the previous code. Also an average is created.
    # to be able to execute parts of the method in isolation, this redundancy was
    # necessary

    import random
    random_int = random.randrange(0, len(x_test))
    min_list = []
    for element in x_test_vectors:
        min_list.append(cosine(element, x_test_vectors[random_int]))
    index_min = np.argmin(min_list)

    indices_reduced = range(len(min_list))
    a, indices_reduced = zip(*sorted(zip(min_list, indices_reduced)))

    from matplotlib import pyplot as plt
    # plot original image as first
    plt.figure(figsize=(20, 4))
    ax = plt.subplot(2, 8, 1)
    plt.imshow(x_test[random_int])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot calculated nearest neighbors
    for i in range(15):
        ax = plt.subplot(2, 8, i + 2)
        plt.imshow(x_test[indices_reduced[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    # the first image is the one we want to find neighbors for, the results are bad
    # results with other methods are much better and are returning brown houses"""


"""
    Function displaying nearest neighbors calculated with brute force

    Attributes
    ----------
    x_final_vectors: array
        vectors of images neighbors are supposed to be searched for
    x_final: array
        image array of images neighbors are supposed to be searched for
    random_int: number
        random number referring to the query image
"""
def visualize_brute_force(x_final_vectors, x_final, random_int):
    min_list = []
    for element in x_final_vectors:
        min_list.append(cosine(element,x_final_vectors[random_int]))

    indices_reduced = range(len(min_list))
    a, indices_reduced = zip( *sorted( zip(min_list, indices_reduced)))

    from matplotlib import pyplot as plt
    # plot original image as first
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(2, 6, 1)
    plt.imshow(x_final[random_int]/255)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot calculated nearest neighbors
    for i in range(10):
        ax = plt.subplot(2, 6, i+2)
        plt.imshow(x_final[indices_reduced[i]]/255)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    
    """
    Function displaying nearest neighbors calculated with brute force

    Attributes
    ----------
    num_neighbors: number
        number of neighbors that should be regarded during search
    x_final: array
        image array of images neighbors are supposed to be searched for
    random_int: number
        random number referring to the query image
    index: index
        faiss index
    x_final_vectors: array
        vectors of images neighbors are supposed to be searched for
"""
def visualize_faiss(num_neighbors, x_final, random_int, index, x_final_vectors):
    D, I = index.search(x_final_vectors[random_int:random_int+1], num_neighbors)

    # plot original image as first
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(2, 6, 1)
    plt.imshow(x_final[random_int]/255)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot calculated nearest neighbors
    for i, neighbour in enumerate(I[0][0:11]):
        ax = plt.subplot(2, 6, i+2)
        plt.imshow(x_final[neighbour]/255)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()