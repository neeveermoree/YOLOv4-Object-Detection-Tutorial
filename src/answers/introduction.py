import numpy as np


def transform_coordinates(arr):
    """
    Transform coordinates from (xmin, xmax, ymin, ymax) to (Xc, Yc, width, height)
    Input:
        arr: np.array - shape is (n, 4) where n is amount of bounding boxes and 4 relates to coordinates
    Output
        transformed_arr: np.array - shape is (n, 4)
    """
    
    width = arr[:, 1] - arr[:, 0]
    height = arr[:, 3] - arr[:, 2]
    Xc = arr[:, 0] + width / 2
    Yc = arr[:, 2] + height / 2

    transformed_arr = np.concatenate([Xc, Yc, width, height], axis=1)
    return transformed_arr
