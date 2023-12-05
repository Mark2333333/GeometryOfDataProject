from scipy.spatial.distance import directed_hausdorff
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, euclidean
# from fastdtw import fastdtw

class image_contour:
    contour_coord:[]
    image_index:int
    image_class:str


#hausdorff_distance
def modified_hausdorff_distance(coord_seq1, coord_seq2):
    max_distances = []

    for coord_seq_a, coord_seq_b in [(coord_seq1, coord_seq2), (coord_seq2, coord_seq1)]:
        distances = []
        for point_a in coord_seq_a:
            # Calculate the distance from point_a to each point in coord_seq_b
            # and take the minimum of these distances
            min_distance = min(np.linalg.norm(np.array(point_a) - np.array(point_b)) for point_b in coord_seq_b)
            distances.append(min_distance)
        # Find the maximum of these minimum distances for the sequence
        max_distances.append(max(distances))

    # Return the average of the maximum distances from each sequence
    return np.mean(max_distances)


# examples_h_dis
# contour1 = [(1, 2), (2, 3), (3, 4), (4, 5)]
# contour2 = [(1, 2), (2, 3), (4, 5), (5, 6)]

# distance = modified_hausdorff_distance(contour1, contour2)
# print("Modified Hausdorff Distance:", distance)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def c_frechet_distance(p, q, len_p, len_q, memo):
    if memo[len_p, len_q] > -1:
        return memo[len_p, len_q]
    elif len_p == 0 and len_q == 0:
        memo[len_p, len_q] = euclidean_distance(p[0], q[0])
    elif len_p == 0:
        memo[len_p, len_q] = max(c_frechet_distance(p, q, len_p, len_q-1, memo), euclidean_distance(p[0], q[len_q]))
    elif len_q == 0:
        memo[len_p, len_q] = max(c_frechet_distance(p, q, len_p-1, len_q, memo), euclidean_distance(p[len_p], q[0]))
    else:
        memo[len_p, len_q] = max(
            min(c_frechet_distance(p, q, len_p-1, len_q, memo),
                c_frechet_distance(p, q, len_p, len_q-1, memo),
                c_frechet_distance(p, q, len_p-1, len_q-1, memo)),
            euclidean_distance(p[len_p], q[len_q])
        )
    return memo[len_p, len_q]

#frechet_distance
def frechet_distance(p, q):
    len_p, len_q = len(p) - 1, len(q) - 1  # Adjust indices for 0-based indexing
    memo = np.full((len_p + 1, len_q + 1), -1.0)
    return c_frechet_distance(p, q, len_p, len_q, memo)

# examples_fre_dis：
# contour1 = [(0, 0), (1, 1), (2, 2), (3, 3)]
# contour2 = [(0, 1), (1, 2), (2, 3), (3, 4)]

# frechet_dist = frechet_distance(contour1, contour2)
# print("Frechet Distance:", frechet_dist)

#shape_context_distance
def compute_shape_context(points, nbins_r, nbins_theta):
        alpha=0.1
        beta=1.0
        r = np.sqrt(np.sum(points**2, axis=1))
        theta = np.arctan2(points[:, 1], points[:, 0])
        
        log_r = np.logspace(0, np.log10(2), nbins_r)
        log_theta = np.linspace(0, 2 * np.pi, nbins_theta)

        context = np.zeros((nbins_r, nbins_theta), dtype=np.float)

        for i in range(nbins_r):
            for j in range(nbins_theta):
                d_r = np.abs(r - log_r[i])
                d_theta = np.abs(theta - log_theta[j])
                context[i, j] = np.sum(np.exp(-alpha * d_r**2 - beta * d_theta**2))

        return context / np.sum(context)

#dtw_distance
def dtw_distance(series1, series2):
    x1, y1 = zip(*series1)
    x2, y2 = zip(*series2)

    distance_matrix = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            distance_matrix[i, j] = euclidean((x1[i], y1[i]), (x2[j], y2[j]))

    _, dtw_dist = fastdtw(distance_matrix)

    return dtw_dist

# extract test_file
def test(file_path):
    df=pd.read_csv(file_path)
    grouped_data = df.groupby('indexName')
    list_of_dicts = [
    {
        'indexName': index_name,
        'class': list(group['genusName'])[0],
        'coordinates': [[row['X'], row['Y']] for _, row in group.iterrows()]
    } for index_name, group in grouped_data
    ]
    return list_of_dicts

# dicts = test("/Users/tangzhiyan/Downloads/GeoProject/GeometryOfDataProject/Contours_GastropodShells.csv")
# print(frechet_distance(np.array(dicts[0]['coordinates']),np.array(dicts[27]['coordinates'])))







        



