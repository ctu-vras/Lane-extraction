import numpy as np
import pandas as pd
import pickle
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

def dbscan_clustering(data, eps=1, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    return clustering.labels_

def save_colored_segments_to_point_cloud(data, labels, filename='colored_clusters.ply'):

    unique_labels = np.unique(labels)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))

    points_with_colors = []

    for point, label in zip(data, labels):
        color = colors[np.where(unique_labels == label)][0]
        point_with_color = np.concatenate((point, color))
        points_with_colors.append(point_with_color)

    points_with_colors = np.array(points_with_colors)
    points_df = pd.DataFrame(points_with_colors, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
    points_df[['red', 'green', 'blue']] = points_df[['red', 'green', 'blue']].astype(np.uint8)

    point_cloud = PyntCloud(points_df)
    point_cloud.to_file(filename)

def ransac_line_fitting(data, labels, residual_threshold):

    line_models = {}

    for label in np.unique(labels):
        points = data[labels == label]

        if len(points) < 5:
            # Skipping label because there are too little points
            continue

        ransac = RANSACRegressor(residual_threshold=residual_threshold)
        ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])

        coef = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        inlier_mask = ransac.inlier_mask_

        line_models[label] = {'coef': coef, 'intercept': intercept, 'inlier_mask': inlier_mask}

    return line_models

def filter_by_percentage(data, labels, line_models, percentage_threshold):
    filtered_line_models = {}

    for label, value in line_models.items():
        points = data[labels == label]
        num_points_segment = len(points)
        num_points_line = np.sum(value['inlier_mask'])

        # Calculate the percentage of inliers in the line
        percentage = num_points_line / num_points_segment

        # Only keep inlier point indices that meet the percentage threshold
        if percentage >= percentage_threshold:
            filtered_line_models[label] = value

    return filtered_line_models

def extract_points(data, line_models, labels):
    inlier_points = []
    new_labels = []

    for label in line_models:
        inlier_mask = line_models[label]['inlier_mask']

        points = data[labels == label]
        inlier_points.append(points[inlier_mask])

        points_label = labels[labels == label]
        new_labels.append(points_label[inlier_mask])

    return np.vstack(inlier_points), np.concatenate(new_labels, axis=0)

def main():
    filepath = "projection.pcd"
    cloud = PyntCloud.from_file(filepath)
    data = cloud.points
    data_np = data.to_numpy()

    # Apply DBSCAN clustering
    labels = dbscan_clustering(data_np)

    clear_data = data_np[labels != -1]
    labels = labels[labels != -1]

    save_colored_segments_to_point_cloud(clear_data, labels, filename='segmentation/colored_clusters.ply')

    # Apply RANSAC-based line fitting
    lines_models = ransac_line_fitting(clear_data, labels, residual_threshold = 1)

    lines_models = filter_by_percentage(clear_data, labels, lines_models, percentage_threshold=0.5)
    lines_points, lines_labels = extract_points(clear_data, lines_models, labels)

    save_colored_segments_to_point_cloud(lines_points, lines_labels, filename='segmentation/colored_lines.ply')
    
    # save to file
    with open('segmentation/data.pkl', 'wb') as f:
        pickle.dump((lines_models, lines_points, lines_labels), f)

if __name__ == "__main__":
    main()