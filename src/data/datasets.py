# Author: Lennart Seeger
# This file is licensed under the Apache 2.0 license available at
# http://www.apache.org/licenses/LICENSE-2.0
import sys
import numpy as np
from supportive.reshape_images import reshape_images
from sklearn.model_selection import train_test_split

sys.path.insert(1, '../src')

"""
    Function that yields the MLRSNET dataset.

    Attributes
    ----------
    image_size: int
        size to which the images should be resized
    
    Returns
    ----------
    "x_train": array
        training data

    "y_train": array
        training labels
        
    "x_val": array
        validation data

    "y_val": array
        validation labels
        
    "x_test": array
        test data

    "y_test": array
        test labels
"""
def get_mlrsnet(image_size=256):
    train = np.load("../data/mlrsnet.npz")

    x_original = train['images']
    if image_size != 256:
        x_original = reshape_images(x_original, image_size)
    y_original = train['labels']

    y_original[y_original == "airplane"] = 0
    y_original[y_original == "airport"] = 1
    y_original[y_original == "bareland"] = 2
    y_original[y_original == "baseball_diamond"] = 3
    y_original[y_original == "basketball_court"] = 4
    y_original[y_original == "beach"] = 5
    y_original[y_original == "bridge"] = 6
    y_original[y_original == "chaparral"] = 7
    y_original[y_original == "cloud"] = 8
    y_original[y_original == "commercial_area"] = 9
    y_original[y_original == "dense_residential_area"] = 10
    y_original[y_original == "desert"] = 11
    y_original[y_original == "eroded_farmland"] = 12
    y_original[y_original == "farmland"] = 13
    y_original[y_original == "forest"] = 14
    y_original[y_original == "freeway"] = 15
    y_original[y_original == "golf_course"] = 16
    y_original[y_original == "ground_track_field"] = 17
    y_original[y_original == "harbor&port"] = 18
    y_original[y_original == "industrial_area"] = 19
    y_original[y_original == "intersection"] = 20
    y_original[y_original == "island"] = 21
    y_original[y_original == "lake"] = 22
    y_original[y_original == "meadow"] = 23
    y_original[y_original == "mobile_home_park"] = 24
    y_original[y_original == "mountain"] = 25
    y_original[y_original == "overpass"] = 26
    y_original[y_original == "park"] = 27
    y_original[y_original == "parking_lot"] = 28
    y_original[y_original == "parkway"] = 29
    y_original[y_original == "railway"] = 30
    y_original[y_original == "railway_station"] = 31
    y_original[y_original == "river"] = 32
    y_original[y_original == "roundabout"] = 33
    y_original[y_original == "shipping_yard"] = 34
    y_original[y_original == "snowberg"] = 35
    y_original[y_original == "sparse_residential_area"] = 36
    y_original[y_original == "stadium"] = 37
    y_original[y_original == "storage_tank"] = 38
    y_original[y_original == "swimming_pool"] = 39
    y_original[y_original == "swimmimg_pool"] = 39
    y_original[y_original == "tennis_court"] = 40
    y_original[y_original == "terrace"] = 41
    y_original[y_original == "transmission_tower"] = 42
    y_original[y_original == "vegetable_greenhouse"] = 43
    y_original[y_original == "wetland"] = 44
    y_original[y_original == "wind_turbine"] = 45

    y_original = y_original.astype('int64')
    x_train, x_test, y_train, y_test = train_test_split(x_original, y_original,
                                                        test_size=0.005)
    x_test = x_test[0:500]
    y_test = y_test[0:500]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.01)
    x_train = x_train[0:100000]
    y_train = y_train[0:100000]

    return x_train, y_train, x_val, y_val, x_test, y_test


"""
    Function that yields the UCMERCED dataset.

    Attributes
    ----------
    image_size: int
        size to which the images should be resized
    
    Returns
    ----------
    "x_train": array
        training data

    "y_train": array
        training labels
        
    "x_val": array
        validation data

    "y_val": array
        validation labels
        
    "x_test": array
        test data

    "y_test": array
        test labels
"""
def get_ucmerced(image_size=256, false_labels=True):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    train = np.load("../data/ucmerced.npz")

    x_original = train['images']
    x_original = reshape_images(x_original, image_size)
    y_original = train['labels']

    y_original[y_original == "agricultural"] = 0
    y_original[y_original == "airplane"] = 1
    y_original[y_original == "baseballdiamond"] = 2
    y_original[y_original == "beach"] = 3
    y_original[y_original == "buildings"] = 4
    y_original[y_original == "chaparral"] = 5
    y_original[y_original == "denseresidential"] = 6
    y_original[y_original == "forest"] = 7
    y_original[y_original == "freeway"] = 8
    y_original[y_original == "golfcourse"] = 9
    y_original[y_original == "harbor"] = 10
    y_original[y_original == "intersection"] = 11
    y_original[y_original == "mediumresidential"] = 12
    y_original[y_original == "mobilehomepark"] = 13
    y_original[y_original == "overpass"] = 14
    y_original[y_original == "parkinglot"] = 15
    y_original[y_original == "river"] = 16
    y_original[y_original == "runway"] = 17
    y_original[y_original == "sparseresidential"] = 18
    y_original[y_original == "storagetanks"] = 19
    y_original[y_original == "tenniscourt"] = 20

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    y_original = y_original.astype('int64')
    x_train, x_test, y_train, y_test = train_test_split(x_original, y_original,
                                                        test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    if false_labels:
        y_train = np.array([-1] * len(y_train))
        y_val = np.array([-1] * len(y_val))

    return x_train, y_train, x_val, y_val, x_test, y_test


"""
    Function that yields the Denmark testset.

    Attributes
    ----------
    image_size: int
        size to which the images should be resized
    
    Returns
    ----------
    "x_test": array
        test data

    "y_test": array
        test labels
"""
def get_denmark(image_size=256):
    train = np.load("../data/denmark.npz")

    x_original = train['images']
    x_original = reshape_images(x_original, image_size)
    y_original = train['labels']

    y_original[y_original == "agricultural_brown"] = 0
    y_original[y_original == "agricultural_green"] = 1
    y_original[y_original == "big_buildings"] = 2
    y_original[y_original == "brown_buildings"] = 3
    y_original[y_original == "coast"] = 4
    y_original[y_original == "curve"] = 5
    y_original[y_original == "forest"] = 6
    y_original[y_original == "grey_buildings"] = 7
    y_original[y_original == "harbor"] = 8
    y_original[y_original == "highway"] = 9
    y_original[y_original == "lake"] = 10
    y_original[y_original == "parking_lot_crooked"] = 11
    y_original[y_original == "parking_lot_straight"] = 12
    y_original[y_original == "rail"] = 13
    y_original[y_original == "river"] = 14
    y_original[y_original == "road_buildings"] = 15
    y_original[y_original == "round_water"] = 16
    y_original[y_original == "silo_big"] = 17
    y_original[y_original == "silo_small"] = 18
    y_original[y_original == "single_road"] = 19
    y_original[y_original == "solar"] = 20
    y_original[y_original == "t_crossing"] = 21
    y_original[y_original == "tree_line"] = 22
    y_original[y_original == "water"] = 23
    y_original[y_original == "windmill"] = 24

    y_original = y_original.astype('int64')
    x_test = x_original
    y_test = y_original
    return x_test, y_test
