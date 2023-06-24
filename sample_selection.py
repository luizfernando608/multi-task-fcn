from osgeo import gdal

import os
import numpy as np


from skimage.measure import label, regionprops_table

from src.utils import read_tiff, load_norm, array2raster, read_yaml
from src.metrics import evaluate_metrics, evaluate_metrics_pred

import pandas as pd

import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler


def undersample(data:pd.DataFrame, column_category:str):

    X = data.drop(column_category, axis=1).copy()
    
    X.reset_index(inplace=True,names=["index"] )

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
    


def get_components_stats(components_img: np.array, label_img: np.array):
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
    component_ids_to_remove: np.array,
    components_img: np.array,
    label_img: np.array,
    component_stats: pd.DataFrame,
):

    for idx in component_ids_to_remove:
        label_img[components_img == idx] = 0

        components_img[components_img == idx] = 0

    # remove from dataframe
    component_stats.drop(component_ids_to_remove, inplace=True)



def get_labels_delta( old_components_img:np.array, new_components_img:np.array, new_label_img:np.array)->np.array:
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
    ### CREATE NEW SET ###
    # all_label_set = ground_truth_img.copy()

    label_delta = np.zeros_like(new_components_img)

    components_to_iter = np.unique(new_components_img)
    components_to_iter = components_to_iter[components_to_iter!=0]

    for idx in components_to_iter:
        # se mais de 90% do componente é vazio será adicionado a nova amostra predita
        if np.mean(old_components_img[new_components_img==idx] == 0) > 0.9:
            # count labels
            unique_labels, count_labels = np.unique(new_label_img[new_components_img==idx], return_counts=True)
                
            # remove 0 count
            unique_labels = unique_labels[unique_labels!=0]
            count_labels = count_labels[unique_labels!=0]

            # get valeu of class with higher count
            class_common_idx = np.argmax(count_labels)
            class_common = unique_labels[class_common_idx]
            
            # all_label_set[components_pred_map==idx] = class_common
            label_delta[new_components_img==idx] = class_common
    
    return label_delta



def get_new_segmentation_sample(ground_truth_img, pred_map, prob_map):
    # set labels at the same scale as ground truth labels
    pred_map += 1

    pred_map_99 = np.where(prob_map > 0.99, pred_map, 0)

    components_pred_map = label(pred_map_99)
    components_gt_label = label(ground_truth_img)

    stats_pred_data = get_components_stats(components_pred_map, pred_map_99)
    stats_gt_data = get_components_stats(components_gt_label, ground_truth_img)

    # FILTER AREA WITH LESS THAN 200 pixels of area
    filter_area = stats_pred_data["area"] < 200

    component_ids_to_remove = np.array(stats_pred_data[filter_area].index.astype(int))

    remove_components_by_index(
        component_ids_to_remove = component_ids_to_remove,
        components_img = components_pred_map,
        label_img = pred_map_99,
        component_stats = stats_pred_data,
    )


    # FILTER BY CONVEX/AREA
    filter_ratio_convex_area = (stats_pred_data["area"]/stats_pred_data["convex_area"]) < 0.85

    component_ids_to_remove = np.array(stats_pred_data[filter_ratio_convex_area].index.astype(int))
    
    remove_components_by_index(
        component_ids_to_remove,
        components_img = components_pred_map,
        label_img = pred_map_99,
        component_stats = stats_pred_data
    )

    # get the new components that are not in the old components
    delta_gt_pred_labels = get_labels_delta(old_components_img=components_gt_label, new_components_img=components_pred_map, new_label_img=pred_map_99)
    components_delta = label(delta_gt_pred_labels)
    
    # join all labels set 
    all_labes_set = np.where(components_delta!=0, delta_gt_pred_labels, ground_truth_img)


    # Select balanced sample
    stats_delta = get_components_stats(components_img=components_delta, label_img=delta_gt_pred_labels)
    
    stats_und_label_delta = undersample(stats_delta, "tree_type")

    id_selected_components = np.array(stats_und_label_delta.index, dtype=int)

    selected_labels_set = np.where(np.isin(components_delta, id_selected_components), delta_gt_pred_labels, ground_truth_img)
    
    
    return selected_labels_set, all_labes_set, delta_gt_pred_labels



if __name__ == "__main__":
    args = read_yaml("args.yaml")

    # data parameters
    itc = False
    sum_overlap = 1.1

    # paths to the files from one iteration
    current_iter_folder = "/home/luiz/multi-task-fcn/MyData/iter_1"
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

    get_new_segmentation_sample(ground_truth_img, pred, prob)

    print("Hello World!")
