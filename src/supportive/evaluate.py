# Author: Lennart Seeger
# This file is licensed under the Apache 2.0 license available at
# http://www.apache.org/licenses/LICENSE-2.0
import numpy as np
import sys
from scipy.spatial.distance import cosine

sys.path.insert(1, '../src')

"""
    Function calculating the custom neighbor accuracy

    Attributes
    ----------
    method: method
        method that should be used for feature extraction
    x_test: array
        test dataset
    y_test: array
        labels of test dataset
    neighbors: number
        number of neighbors that should be regarded
    
    Returns
    ----------
    : number
        accuracy of extractor
"""
def evaluate_extractor(extractor, x_test, y_test, neighbors=10):
    test_vec = extractor(x_test)

    counter = 0
    for elem_idx in range(0, len(x_test)):
        min_list = []
        for element in test_vec:
            min_list.append(cosine(element, test_vec[elem_idx]))

        indices_red = range(len(min_list))
        a, indices_red = zip(*sorted(zip(min_list, indices_red)))

        for index in indices_red[0:neighbors]:
            if np.array_equal(y_test[index], y_test[elem_idx]):
                counter += 1

    return (counter / neighbors / len(x_test))


"""
    Function calculating the custom neighbor accuracy for each class

    Attributes
    ----------
    method: method
        method that should be used for feature extraction
    x_test: array
        test dataset
    y_test: array
        labels of test dataset
    classes: array
        name of the classes
    classes: number
        numberclasses
    neighbors: number
        number of neighbors that should be regarded
"""
def evaluate_extractor_classes(method, x_test, y_test, classesnames, neighbors=10,classes=25):
    x_test_vectors=method(x_test)
    
    for i in range(classes):
        counter_class=0
        for element_idx in (np.array(range(0,500))[y_test==i]):
            counter = 0

            # automated multiple executions of the previous code. Also an average is created.
            # to be able to execute parts of the method in isolation, this redundancy was necessary
            min_list = []
            for element in x_test_vectors:
                min_list.append(cosine(element,x_test_vectors[element_idx]))
            indices_reduced = range(len(min_list))
            a, indices_reduced = zip( *sorted( zip(min_list, indices_reduced) ) )
            len(indices_reduced[0:neighbors])

            for index in indices_reduced[0:neighbors]:
                if np.array_equal(y_test[index], y_test[element_idx]):
                    counter += 1
            counter_class += counter

        print(classesnames[i], counter_class/neighbors/20)
        
"""
    Function calculating which classes were predicted for a certain query class

    Attributes
    ----------
    method: method
        method that should be used for feature extraction
    x_test: array
        test dataset
    y_test: array
        labels of test dataset
    classes: array
        name of the classes
    classes: number
        numberclasses
    neighbors: number
        number of neighbors that should be regarded
    
    Returns
    ----------
    matrix : array
        predicted classes referring to a query class
"""
def determine_predicted_classes(method, x_test, y_test, classesnames, neighbors=10,classes=25):
    x_test_vectors=method(x_test)
    neighbour_dist_array=np.zeros(500).reshape(1,500)
    for p in range(len(x_test_vectors)):
        neighbour_dist_array_single=np.zeros(1)
        for element in x_test_vectors:
            neighbour_dist_array_single=np.append(neighbour_dist_array_single,(cosine(element,x_test_vectors[p])))
        neighbour_dist_array_single=neighbour_dist_array_single[1:].reshape(1,500)
        neighbour_dist_array=np.append(neighbour_dist_array,(neighbour_dist_array_single), axis=0)
    neighbour_dist_array=neighbour_dist_array[1:]
    
    matrix=np.zeros(25).reshape(1,25)
    for i in range(classes):
        counter_class=0
        print("-----",classesnames[i],"-----")
        row=np.zeros(1)
        for j in range(classes):
            counter_class_pred=0
            for element_idx in (np.array(range(0,500))[y_test==i]):
                counter = 0
                min_list=neighbour_dist_array[element_idx]

                indices_reduced = range(len(min_list))
                a, indices_reduced = zip( *sorted( zip(min_list, indices_reduced) ) )
                len(indices_reduced[0:neighbors])

                for index in indices_reduced[0:neighbors]:
                    if np.array_equal(y_test[index], y_test[element_idx]):
                        counter += 1
                    if np.array_equal(j, y_test[index]):
                        counter_class_pred += 1
                counter_class += counter
            row=np.append(row,counter_class_pred)
        print(classesnames[i], (counter_class/neighbors/20/25))
        row=row[1:].reshape(1,25)
        matrix=np.append(matrix,row,axis=0)
    matrix=matrix[1:]
    return matrix