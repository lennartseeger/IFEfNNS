# Author: Lennart Seeger, PSEFSD
# This file is licensed under the Apache 2.0 license available at
# http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np
import rasterio as rio
import math
import itertools
from more_itertools import ichunked
from pyproj import Transformer

sys.path.insert(1, '../src')


"""
    Generator function that iterates over a .tif file and
     returns all data windows for the specified iteration parameters.

    Attributes
    ----------
    cog: string
        filepath to underlying .tif

    size: int
        window size

    start_row: int
        starting pixel row offset (aka top left pixel, row value) of
        the first desired window

    start_col: int
        starting pixel column offset (aka top left pixel, col value) of
        the first desired window

    step_jump: float
        factor of size to be used as step size, for example size=100:
        step_jump=1 -> step size 100, step factor=2 -> step size 200.
        Resulting product must must be an integer or will be rounded up. 
        step_jump < 0 will default to step_jump=1.

    batch_size: int
        size of batches to be yielded. Must be 1 or greater, else
        defaults to 1.

    out_crs: int
        output coordinates can optionally be transformed to this EPSG codes CRS
    
        return:

    Yields lists of data windows of length batch_size.
    A single data window is a dict with the following keys:

    "row": int
        row offset value

    "col": int
        column offset value

    "coordinates": Tuple[float, float]
        Tuple containing the X,Y coordinates of the center pixel

    "size": int
        window size

    "window": Array[Array[Array[int]]]
        raw image data
"""
def batch_generator_denmark(cog="/home/jovyan/work/satellite_data/all_cog.tif", size=256,
                            batch_size=1000, start_row=0, start_col=0, out_crs=3857,
                            step_jump=1):
    with rio.open(cog) as dataset:
        height = dataset.height
        width = dataset.width
        print(height)
        print(width)
        # initialize transformer for later use
        transformer = Transformer.from_crs(
            dataset.crs.to_epsg(), out_crs, always_xy=True)

    # validate step jump value
    if step_jump < 0:
        step_jump = 1
    step = size * step_jump
    if step % 1 > 0:
        step = math.ceil(step)

    # build iterator ranges for row/col offsets
    steps_col = math.ceil(width / step)
    steps_row = math.ceil(height / step)
    range_rows = range(0, steps_row * step, step)
    range_cols = range(0, steps_col * step, step)

    # product of ranges creates full map iterator
    range_product = itertools.product(range_rows, range_cols)

    # if offsets are specified, this drops from the iterator while they are not reached
    range_final = itertools.dropwhile(
        lambda x: x[0] < start_row or x[1] < start_col, range_product
    )

    # split into batches to limit returned lines
    if batch_size < 1:
        batch_size = 1
    batches = ichunked(range_final, batch_size)

    with rio.open(cog) as dataset:

        for batch in batches:
            # empty window list
            windows = []

            # read window, get coordinates, append to windows w/ relevant info
            for row, col in batch:
                window = dataset.read(window=((row, row + size), (col, col + size)))
                (x_coord, y_coord) = dataset.xy(row + (size // 2), col + (size // 2))
                coordinates = transformer.transform(x_coord, y_coord)
                windows.append(
                    {
                        "row": row,
                        "col": col,
                        "coordinates": coordinates,
                        "size": size,
                        "window": window,
                    }
                )

            # yield list of windows for batch
            test_batch = windows
            pixels = [(elem['row'], elem['col']) for elem in test_batch if
                      np.count_nonzero(elem['window']) != 0 and elem['window'].std() != 0]

            yield (pixels)

        
# no longer needed, custom batch function if custom changes are supposed to be made to data during training
def batch_generator(X, Y, batch_size):
    indices = np.arange(len(X))
    batch = []
    while True:
        # it might be a good idea to shuffle your data before each epoch"""*2"""
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                yield X[batch], Y[batch]
                batch = []