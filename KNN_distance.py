from scipy.spatial.distance import directed_hausdorff
import pandas as pd
import numpy as np

class image_contour:
    contour_coord:[]
    image_index:int
    image_class:str


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
contour1 = [(1, 2), (2, 3), (3, 4), (4, 5)]
contour2 = [(1, 2), (2, 3), (4, 5), (5, 6)]

distance = modified_hausdorff_distance(contour1, contour2)
print("Modified Hausdorff Distance:", distance)

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

def frechet_distance(p, q):
    len_p, len_q = len(p) - 1, len(q) - 1  # Adjust indices for 0-based indexing
    memo = np.full((len_p + 1, len_q + 1), -1.0)
    return c_frechet_distance(p, q, len_p, len_q, memo)

# examples_fre_dis：
contour1 = [(0, 0), (1, 1), (2, 2), (3, 3)]
contour2 = [(0, 1), (1, 2), (2, 3), (3, 4)]

frechet_dist = frechet_distance(contour1, contour2)
print("Frechet Distance:", frechet_dist)

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

dicts = test("/Users/tangzhiyan/Downloads/GeoProject/GeometryOfDataProject/Contours_GastropodShells.csv")
print(frechet_distance(np.array(dicts[0]['coordinates']),np.array(dicts[27]['coordinates'])))







        



