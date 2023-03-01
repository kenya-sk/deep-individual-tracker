import numpy as np
from detector.clustering import apply_clustering_to_density_map


def test_apply_clustering_to_density_map():
    # check empty array
    density_map = np.zeros((214, 214))
    centroid_array = apply_clustering_to_density_map(density_map, 10, thresh=0.5)
    assert centroid_array.shape == (0, 2)
    assert centroid_array.sum() == 0

    # check threshold
    density_map = np.zeros((214, 214))
    density_map[100, 100] = 0.4
    density_map[101, 101] = 0.6
    centroid_array = apply_clustering_to_density_map(density_map, 10, thresh=0.5)
    assert centroid_array.shape == (1, 2)
    assert centroid_array.dtype == np.int32

    # check single cluster
    density_map = np.zeros((214, 214))
    density_map[100, 100] = 1
    centroid_array = apply_clustering_to_density_map(density_map, 10, thresh=0.5)
    assert centroid_array.shape == (1, 2)
    assert centroid_array.dtype == np.int32
    assert centroid_array[0, 0] == 100
    assert centroid_array[0, 1] == 100

    # check multiple clusters
    density_map = np.zeros((214, 214))
    density_map[100, 100] = 1
    density_map[101, 101] = 1
    density_map[100, 102] = 1
    density_map[200, 150] = 1
    density_map[203, 151] = 1
    centroid_array = apply_clustering_to_density_map(density_map, 10, thresh=0.5)
    assert centroid_array.shape == (2, 2)
    assert centroid_array.dtype == np.int32
    assert 100 <= centroid_array[0, 0] <= 102
    assert 100 <= centroid_array[0, 1] <= 102
    assert 150 <= centroid_array[1, 0] <= 151
    assert 200 <= centroid_array[1, 1] <= 203
