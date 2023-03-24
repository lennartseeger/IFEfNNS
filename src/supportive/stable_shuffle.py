# Author: PSEFSD
# This file is licensed under the Apache 2.0 license available at
# http://www.apache.org/licenses/LICENSE-2.0
import numpy as np


"""
    Function to shuffle to arrays stable, meaning that the same permutation is applied to both

    Attributes
    ----------
    array_a: array
        array one for shuffeling
    array_b: array
        array two for shuffeling
    
    Returns
    ----------
    : array
        shuffled array one
    : array
        shuffled array two
"""
def stable_shuffle(array_a, array_b):
    assert len(array_a) == len(array_b)
    random_array = np.random.permutation(len(array_a))
    return array_a[random_array], array_b[random_array]