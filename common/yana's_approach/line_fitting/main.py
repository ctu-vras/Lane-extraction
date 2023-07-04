import pickle
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.distance import euclidean

class Line:
 def __init__(self, coef, intercept, points, type_of_line):
    self.coef = coef
    self.intercept = intercept
    self.points = points
    self.points_x = points[:, 0]
    self.points_y = self.coef * self.points_x + self.intercept
    self.type = type_of_line
    self.point1, self.point2 = self.end_points()

 def end_points(self):
    """Calculate the end points of a line based on the points on the line"""
    min_i = np.argmin(self.points_x)
    max_i = np.argmax(self.points_x)
    x_min, y_min = self.points_x[min_i], self.points_y[min_i]
    x_max, y_max = self.points_x[max_i], self.points_y[max_i]
    return [(x_min, y_min), (x_max, y_max)]

def areConnected(line1, line2, threshold):
    """Check if two lines are connected based on a minimum threshold"""
    min_distance = float('inf')
    max_distance = float('-inf')
    min_pair = (None, None)
    max_pair = (None, None)

    for a in line1.point1, line1.point2:
        for b in line2.point1, line2.point2:
            distance = euclidean(a[:2], b[:2])
            if distance < min_distance:
                min_distance = distance
                min_pair = (a, b)
            if distance > max_distance:
                max_distance = distance
                max_pair = (a, b)

    inliers_line2 = find_inlier_points(line1.coef, line1.intercept, min_pair[0][0], line2.points, growth_factor=1)
    nearest_points2 = [point for point in line2.points if euclidean(min_pair[1], point[:2]) < 5]
    inliers_line1 = find_inlier_points(line2.coef, line2.intercept, min_pair[1][0], line1.points, growth_factor=1)
    nearest_points1 = [point for point in line1.points if euclidean(min_pair[0], point[:2]) < 5]
    return (np.any(np.isin(nearest_points2, inliers_line2)) or np.any(np.isin(nearest_points1, inliers_line1))) and min_distance < threshold

def find_inlier_points(line_coeff, line_intercept, start_x, points, growth_factor):
    """Find the inlier points on a line using the equation of a line"""
    inliers = []

    for point in points:
        x, y = point[:2]

        # Calculate expected y-value based on the line's equation
        expected_y = line_coeff * x + line_intercept

        # Calculate the threshold, which starts at 0 for x = start_x and then grows proportionally to the distance from start_x
        threshold = growth_factor * abs(x - start_x)

        # If the actual y-value is within the threshold of the expected y-value, it's an inlier
        if abs(y - expected_y) <= threshold:
            inliers.append(point)

    return np.array(inliers)

def get_farthest_lines(lines):
    """Get the two farthest lines among a list of lines"""
    if len(lines) < 2:
        return lines
    max_distance = -1
    line1, line2 = None, None
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            # Calculate distances for all four point pairs
            distances = [
                euclidean(lines[i].point1, lines[j].point1),
                euclidean(lines[i].point1, lines[j].point2),
                euclidean(lines[i].point2, lines[j].point1),
                euclidean(lines[i].point2, lines[j].point2)
            ]
            # If the max distance of this pair is greater than the current max_distance, update the lines and max_distance
            pair_max_distance = max(distances)
            if pair_max_distance > max_distance:
                max_distance = pair_max_distance
                line1, line2 = lines[i], lines[j]
    return [line1, line2]

def add_line(all_lines, line, threshold_angle = 1.5, threshold_intercept = 10):
    """Add a line to the list of all lines if it fits the criteria"""
    if line.points.shape[0] < 1:
        return
    for key, value in all_lines.items():
        for line_ in get_farthest_lines(value):
            if abs(line.coef - line_.coef) < threshold_angle and areConnected(line, line_, threshold_intercept):
                all_lines[key].append(line)
                return all_lines
    all_lines[len(all_lines)] = [line]
    return all_lines

def fit_and_categorize_lines(lines_models, lines_points, lines_labels):
    """
    Fit and categorize lines using a RANSAC regressor and based on their proximity to each other.
    
    Parameters:
    lines_models : dict
        Dictionary containing line coefficients and other properties.
    lines_points : np.array
        Numpy array of points in the lines.
    lines_labels : np.array
        Numpy array of labels corresponding to the lines.

    Returns:
    all_fitted_lines : dict
        Dictionary of all fitted lines.
    """

    # Define the threshold for considering whether two lines are separate or part of a dashed line
    threshold_for_dashed = 7
    
    # Initialize the RANSAC regressor
    ransac = RANSACRegressor(residual_threshold=1)
    
    # Initialize a dictionary to hold all the fitted lines
    all_fitted_lines = {}

    # Continue fitting lines until no points remain
    while lines_points.shape[0] > 0:

        # Fit a line to the points
        ransac.fit(lines_points[:, 0].reshape(-1, 1), lines_points[:, 1])
        
        # Get the line's coefficient and intercept
        coef = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_

        # Identify the inliers, i.e., points that fall on the fitted line
        inlier_mask = ransac.inlier_mask_
        
        # Get labels of the inliers
        inlier_labels = lines_labels[inlier_mask]
        
        # Identify unique labels and their counts
        unique_labels, counts_labels = np.unique(inlier_labels, return_counts=True)

        # Identify the first inlier label
        first_inlier_label = unique_labels[0]

        # Iterate over the unique labels
        for label in unique_labels:
            # If the difference in coefficients is more than 1, discard the corresponding points from inliers
            if abs(lines_models[label]['coef'] - coef) > 1:
                inlier_mask[lines_labels == label] = False

        # Update the inliers and labels
        inlier_labels = lines_labels[inlier_mask]
        unique_labels, counts_labels = np.unique(inlier_labels, return_counts=True)

        # Update the inlier points based on the updated inlier mask
        if len(unique_labels) > 0:
            inlier_points = lines_points[inlier_mask]
        else:
            inlier_points = lines_points[lines_labels == first_inlier_label]
            inlier_labels = first_inlier_label
            unique_labels, counts_labels = np.unique(inlier_labels, return_counts=True)

        # If there's only one unique label, add a 'full' line
        if len(unique_labels) == 1:
            all_fitted_lines = add_line(all_fitted_lines, line=Line(coef, intercept, inlier_points, "full"))

        else:
            # Compute the mean of each cluster
            cluster_means = np.array([inlier_points[inlier_labels == label].mean(axis=0) for label in unique_labels])

            # Sort clusters by the x-coordinate of their means
            sorted_indices = np.argsort(cluster_means[:, 0])
            sorted_means = cluster_means[sorted_indices]
            sorted_labels = unique_labels[sorted_indices]

            # Compute Euclidean distances between neighboring clusters
            cluster_distances = np.sqrt((np.diff(sorted_means, axis=0)**2).sum(axis=1))

            # Identify clusters where all neighboring distances are above the threshold
            normal_cluster_masks = np.concatenate(([cluster_distances[0] < threshold_for_dashed], (cluster_distances[:-1] < threshold_for_dashed) & (cluster_distances[1:] < threshold_for_dashed), [cluster_distances[-1] < threshold_for_dashed]))
            normal_labels = sorted_labels[normal_cluster_masks]

            # Identify clusters that don't meet the distance threshold
            wrong_cluster_masks = np.logical_not(normal_cluster_masks)
            wrong_labels = sorted_labels[wrong_cluster_masks]

            # Add lines to the 'all_fitted_lines' dictionary based on the condition
            if len(unique_labels) == np.sum(normal_cluster_masks):
                all_fitted_lines = add_line(all_fitted_lines, line=Line(coef, intercept, inlier_points, "dashed"))
                
            elif np.sum(normal_cluster_masks) == 0:
                most_frequent_label = unique_labels[np.argmax(counts_labels)]
                inlier_points = inlier_points[inlier_labels == most_frequent_label]
                inlier_labels = inlier_labels[inlier_labels == most_frequent_label]
                all_fitted_lines = add_line(all_fitted_lines, line=Line(coef, intercept, inlier_points, "full"))
                
            else:
                # Handling for the case when only part of the labels meet the distance threshold
                normal_clusters_array = np.array(normal_cluster_masks)
                start_normal_sequence = np.argmax(normal_clusters_array)
                end_normal_sequence = start_normal_sequence + np.argmax(~normal_clusters_array[start_normal_sequence:]) - 1

                new_cluster_masks = np.full(len(normal_clusters_array), False)
                if end_normal_sequence < start_normal_sequence:
                    new_cluster_masks[start_normal_sequence:] = True
                else:
                    new_cluster_masks[start_normal_sequence:end_normal_sequence+1] = True
                
                normal_labels = sorted_labels[new_cluster_masks]

                keep_indices = np.isin(inlier_labels, normal_labels)
                inlier_points = inlier_points[keep_indices]
                inlier_labels = inlier_labels[keep_indices]
                all_fitted_lines = add_line(all_fitted_lines, line=Line(coef, intercept, inlier_points, "dashed"))

        # Remove the points and labels of the lines that have been fitted
        processed_mask = np.isin(lines_labels, inlier_labels)

        lines_points = lines_points[np.logical_not(processed_mask)]
        lines_labels = lines_labels[np.logical_not(processed_mask)]
        
    return all_fitted_lines


def main():
    with open('segmentation/data.pkl', 'rb') as f:
        lines_models, lines_points, lines_labels = pickle.load(f)

    all_lines = fit_and_categorize_lines(lines_models, lines_points, lines_labels)

    # save to file
    with open('line_fitting/data.pkl', 'wb') as f:
        pickle.dump((all_lines), f)
    

if __name__ == "__main__":
    main()