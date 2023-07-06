from osgeo import gdal

import os
import numpy as np


from skimage.measure import label, regionprops_table

from src.utils import read_tiff, load_norm, array2raster, read_yaml
from src.metrics import evaluate_metrics, evaluate_metrics_pred

import pandas as pd

import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler

from typing import List, Tuple

def undersample(data: pd.DataFrame, column_category: str):
    X = data.drop(column_category, axis=1).copy()

    X.reset_index(inplace=True, names=["index"])

    y = data[column_category].copy()
    # define undersample strategy
    undersample_selector = RandomUnderSampler(
        sampling_strategy="majority", random_state=42
    )
    # fit and apply the transform
    X, y = undersample_selector.fit_resample(X, data[column_category])

    # summarize class distribution
    # columns_namem = X.columns.tolist()+[column_category]
    undersampled_data = pd.concat([X, y], axis=1)
    undersampled_data.set_index("index", inplace=True)

    return undersampled_data


def get_components_stats(components_img: np.ndarray, label_img: np.ndarray):
    properties = [
        "area",
        "convex_area",
        "bbox_area",
        "extent",
        "solidity",
        "eccentricity",
        "orientation",
        "centroid",
        "bbox",
        "label",
        "intensity_mean",
    ]

    components_stats = pd.DataFrame(
        regionprops_table(
            components_img, properties=properties, intensity_image=label_img
        )
    )
    # set component id as index
    components_stats.set_index("label", inplace=True)
    # tree_type
    components_stats.rename(columns={"intensity_mean": "tree_type"}, inplace=True)
    components_stats["tree_type"] = components_stats["tree_type"].astype(int)

    return components_stats


def remove_components_by_index(
    component_ids_to_remove: np.ndarray,
    components_img: np.ndarray,
    label_img: np.ndarray,
    component_stats: pd.DataFrame,
):
    for idx in component_ids_to_remove:
        label_img[components_img == idx] = 0

        components_img[components_img == idx] = 0

    # remove from dataframe
    component_stats.drop(component_ids_to_remove, inplace=True)


def get_labels_delta(
    old_components_img: np.ndarray, new_components_img: np.ndarray, new_label_img: np.ndarray
) -> np.ndarray:
    """Get the components labels that are in the new image but not in the old image

    Parameters
    ----------
    old_components_img : np.array
        Old image map with the components
    new_components_img : np.array
        New image map with more components than the old image
    new_label_img : np.array
        New image map with the label classes (tree type)

    """

    label_delta = np.zeros_like(new_components_img)

    components_to_iter = np.unique(new_components_img)
    components_to_iter = components_to_iter[components_to_iter != 0]

    for idx in components_to_iter:
        # if more than 90% of the component is empty it will be added to the new predicted sample
        if np.mean(old_components_img[new_components_img == idx] == 0) > 0.9:
            # count labels
            unique_labels, count_labels = np.unique(
                new_label_img[new_components_img == idx], return_counts=True
            )

            # remove 0 count
            unique_labels = unique_labels[unique_labels != 0]
            count_labels = count_labels[unique_labels != 0]

            # get valeu of class with higher count
            class_common_idx = np.argmax(count_labels)
            class_common = unique_labels[class_common_idx]

            # all_label_set[components_pred_map==idx] = class_common
            label_delta[new_components_img == idx] = class_common

    return label_delta

def filter_components_by_geometric_properties(components_pred_map: np.ndarray, pred_labels: np.ndarray):
    """Filter components by geometric properties\n
    The filter rules are:
    - Area < 200 pixels
    - Area/Convex Area < 0.85

    Parameters
    ----------
    components_pred_map : np.array
        Predicted components map
    pred_labels : np.array
        Predicted labels map

    """

    stats_pred_data = get_components_stats(components_pred_map, pred_labels)
    
    # FILTER AREA WITH LESS THAN 200 pixels of area
    filter_area = stats_pred_data["area"] < 200

    component_ids_to_remove = stats_pred_data[filter_area].index
    component_ids_to_remove = np.array(component_ids_to_remove, dtype=int)

    remove_components_by_index(
        component_ids_to_remove=component_ids_to_remove,
        components_img=components_pred_map,
        label_img=pred_labels,
        component_stats=stats_pred_data,
    )


    # FILTER BY CONVEX/AREA
    filter_ratio_convex_area = (stats_pred_data["area"] / stats_pred_data["convex_area"]) < 0.85

    component_ids_to_remove = stats_pred_data[filter_ratio_convex_area].index
    component_ids_to_remove = np.array(component_ids_to_remove.astype(int))

    remove_components_by_index(
        component_ids_to_remove,
        components_img=components_pred_map,
        label_img=pred_labels,
        component_stats=stats_pred_data,
    )

def get_selected_labels(delta_components_img:np.ndarray, delta_pred_map:np.ndarray, old_pred_map:np.ndarray)->np.ndarray:
    """Select the best labels from the new components predicted.
    Use undersampling to select the best labels and to avoid unbalanced classes

    Parameters
    ----------
    delta_components_img : np.array
        New components predicted that were not in the old components predicted
    delta_pred_map : np.array
        New labels predicted that were not in the old labels predicted
    old_pred_map : np.array
        Old labels predicted in the last iteration

    Returns
    -------
    np.array
        New labels predicted with the best components and balanced classes
    """
    
    # Select balanced sample
    stats_delta = get_components_stats(components_img=delta_components_img, label_img=delta_pred_map)

    stats_delta["area_by_convex"] = stats_delta["area"] / stats_delta["convex_area"]

    # stats_und_label_delta = undersample(stats_delta, "tree_type")

    # Undersampling selecting the samples
    min_category_num =  stats_delta.groupby("tree_type").size().min()

    # create a score for best quality
    stats_delta["score"] = (stats_delta["area_by_convex"]/stats_delta["area_by_convex"].max()) +\
                            (stats_delta["area"]/stats_delta["area"].max())
    
    stats_delta.sort_values(by="score", ascending=False, inplace=True)

    stats_und_label_delta = stats_delta.groupby("tree_type").head(min_category_num)

    id_selected_components = np.array(stats_und_label_delta.index, dtype=int)

    selected_labels_set = np.where(
        np.isin(delta_components_img, id_selected_components),
        delta_pred_map,
        old_pred_map,
    )

    return selected_labels_set


def get_new_segmentation_sample(old_pred_map:np.ndarray, new_pred_map:np.ndarray, new_prob_map:np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get the new segmentation sample based on the segmentation from the last iteration and the new segmentation prediction set
    
    Parameters
    ----------
    old_pred_map : np.array
        Segmentation map from the last iteration with tree two labels
    new_pred_map : np.array
        New segmentation map with tree type labels
    new_prob_map : np.array
        New segmentation map with confidence/probability at each pixel
    
    Returns
    -------
    all_labes_set : np.array
        New segmentation map with tree type labels with all components with confidence higher than 0.99

    selected_labels_set : np.array
        The same set as all_labels_set but with some filters applied to minimize unbalanced classes problem
    
    delta_labels_set : np.array
        The difference between the new segmentation map and the segmentation map from the last iteration

    """
    
    # set labels at the same scale as ground truth labels
    new_pred_map += 1
    
    # Select only the components with confidence higher than 0.99
    new_pred_99 = np.where(new_prob_map > 0.99, new_pred_map, 0)
    
    old_components_pred_map = label(old_pred_map)

    new_components_pred_map = label(new_pred_99)
    
    # filter components by geometric properties
    filter_components_by_geometric_properties(
        components_pred_map = new_components_pred_map, 
        pred_labels = new_pred_99
    )


    # get the delta between the new components and the components from the last iteration
    delta_old_new_labels = get_labels_delta(
        old_components_img=old_components_pred_map,
        new_components_img=new_components_pred_map,
        new_label_img=new_pred_99,
    )

    components_delta = label(delta_old_new_labels)

    # join all labels set
    all_labes_set = np.where(components_delta != 0, delta_old_new_labels, old_pred_map)

    selected_labels_set = get_selected_labels(
        delta_components_img = components_delta,
        delta_pred_map = delta_old_new_labels,
        old_pred_map = old_pred_map,
    )

    return all_labes_set, selected_labels_set, delta_old_new_labels


if __name__ == "__main__":
    args = read_yaml("args.yaml")

    # data parameters
    itc = False
    sum_overlap = 1.1

    # paths to the files from one iteration
    current_iter_folder = "/home/luiz/multi-task-fcn/MyData/iter_3"
    depth_path = os.path.join(
        current_iter_folder, "raster_prediction", f"depth_itc{itc}_{sum_overlap}.TIF"
    )
    prob_path = os.path.join(
        current_iter_folder,
        "raster_prediction",
        f"join_prob_itc{itc}_{sum_overlap}.TIF",
    )
    pred_path = os.path.join(
        current_iter_folder,
        "raster_prediction",
        f"join_class_itc{itc}_{sum_overlap}.TIF",
    )

    depth = read_tiff(depth_path)
    prob = read_tiff(prob_path)
    pred = read_tiff(pred_path)

    ground_truth_path = os.path.join(args.data_path, args.train_segmentation_file)
    ground_truth_img = read_tiff(ground_truth_path)

    all_labels, new_labels, delta_labels = get_new_segmentation_sample(ground_truth_img, pred, prob)
    
    print("All labels", np.unique(all_labels))
    print("Selected labels", np.unique(new_labels))
    print("Delta labels", np.unique(delta_labels))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(all_labels)
    ax[1].imshow(new_labels)
    ax[2].imshow(delta_labels)
    # set title
    ax[0].set_title("All labels")
    ax[1].set_title("Selected labels")
    ax[2].set_title("Delta labels")

    # # save figure
    plt.savefig("debug_images/sample_selection_test.png", dpi=600)

