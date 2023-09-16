from osgeo import gdal

import os
import numpy as np

from skimage.measure import label, regionprops_table

from src.utils import read_tiff, read_yaml

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

from imblearn.under_sampling import RandomUnderSampler

from typing import List, Tuple

from generate_distance_map import apply_gaussian_distance_map

from tqdm import tqdm


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
):
    # for idx in component_ids_to_remove:
    #     label_img[components_img == idx] = 0
    #     components_img[components_img == idx] = 0
    
    id_selection = np.argwhere(np.isin(components_img, component_ids_to_remove))
    
    components_img[id_selection[:, 0], id_selection[:, 1]] = 0
    label_img[id_selection[:, 0], id_selection[:, 1]] = 0
    


def get_labels_delta(
    old_label_img: np.ndarray, new_label_img: np.ndarray
) -> np.ndarray:
    """Get the components labels that are in the new image but not in the old image

    Parameters
    ----------
    old_label_img : np.array
        New image map with the label classes (tree type)

    new_label_img : np.array
        New image map with the label classes (tree type)

    """
    new_components_img = label(new_components_img)

    old_components_img = label(old_label_img)

    label_delta = np.zeros_like(new_components_img)

    components_to_iter = np.unique(new_components_img)
    components_to_iter = components_to_iter[np.nonzero(components_to_iter)]

    for idx in tqdm(components_to_iter):
        # if more than 90% of the area is empty it will be added to the new predicted sample
        if np.mean(old_components_img[new_components_img == idx] == 0) > 0.9:
            
            # count labels
            unique_labels, count_labels = np.unique(
                new_label_img[(new_components_img == idx) &  (new_label_img != 0)], return_counts=True
            )

            # get value of class with higher count
            class_common_idx = np.argmax(count_labels)
            class_common = unique_labels[class_common_idx]

            label_delta[new_components_img == idx] = class_common

    return label_delta


def get_label_intersection(
    old_label_img: np.ndarray, new_label_img: np.ndarray
)-> np.ndarray:
    """Get the labels from the new_label_img that are in the old_components_img

    Parameters
    ----------
    old_label_img : np.ndarray
        Segmentation predicted by the previous iteration

    new_label_img : np.ndarray
        Segmentation predicted

    Returns
    ---------
        The labels from the new segmentation that are in the old_components_img
    """
    
    old_components_img = label(old_label_img)
    new_components_img = label(new_label_img)

    label_intersection = np.zeros_like(new_components_img)

    components_to_iter = np.unique(new_components_img)
    components_to_iter = components_to_iter[np.nonzero(components_to_iter)]

    for idx in tqdm(components_to_iter):
        # if more than 90% of the area is filled it will be added to the intersection sample
        if np.mean(old_components_img[new_components_img == idx] == 0) < 0.9:
            
            # count labels
            unique_labels, count_labels = np.unique(
                new_label_img[(new_components_img == idx) &  (new_label_img != 0)], return_counts=True
            )

            # get value of class with higher count
            class_common_idx = np.argmax(count_labels)
            class_common = unique_labels[class_common_idx]

            label_intersection[new_components_img == idx] = class_common

    return label_intersection



def filter_components_by_geometric_property(label_img:np.ndarray, low_limit:float, high_limit:float, property = "area"):
    """Filter components by geometric properties
    Using some property about the compoment, this function select the components between the
    low_limit and the high_limit.

    Parameters
    ----------

    """

    components_img = label(label_img)

    stats_label_data = get_components_stats(components_img, label_img).reset_index()

    filter_property = (stats_label_data[property] < low_limit) | (stats_label_data[property] > high_limit)

    component_ids_to_remove = stats_label_data[filter_property]['label'].astype(int).values

    remove_components_by_index(
        component_ids_to_remove = component_ids_to_remove,
        components_img = components_img,
        label_img = label_img
    )



def select_n_labels_by_class(pred_labels:np.ndarray, samples_by_class:int = 5):

    components_pred_map = label(pred_labels)
    
    delta_stats = get_components_stats(components_pred_map, pred_labels)

    # sample components
    selected_samples = delta_stats.groupby("tree_type").head(samples_by_class)

    # component_ids_to_remove = stats_pred_data[filter_area]["label"]

    # remove selected components
    delta_stats.drop(index=selected_samples.index, inplace=True)

    # remove inplace
    remove_components_by_index(
        component_ids_to_remove = delta_stats.index.astype(int),
        components_img = components_pred_map,
        label_img = pred_labels
    )




def filter_components_by_mask(data_path:str, components_pred_map:np.ndarray, pred_map:np.ndarray):
    """Remove labels and components out of the mask.tif area

    Parameters
    ----------
    data_path : str
        Path to the data folder
    components_pred_map : np.ndarray
        The components map from the current iteration. These components are generated by the function label()
    pred_map : np.ndarray
        The labels map from the current iteration.
    """
    
    mask = read_tiff(os.path.join(data_path, "mask.tif"))
    mask = np.where(mask==99, 0, 1)

    for component in np.unique(components_pred_map):
        component_filter = components_pred_map==component
        area_out_mask = np.mean( mask[component_filter])
        
        if area_out_mask >= 0.20:
            pred_map[component_filter] = 0
            components_pred_map[component_filter] = 0



def join_labels_set(high_priority_labels:np.ndarray, low_priority_labels:np.ndarray, overlap_limit:int=0.05) -> np.ndarray:
    """Join the high priority labels with the low priority labels based on the components area
    The join doesnt overlap the component or cut any component by the other
    If there is an overlap, the component from the high priority labels is kept

    Parameters
    ----------
    high_priority_labels : np.ndarray
        The image arrary with the segmentation labels with high priority.

    low_priority_labels : np.ndarray
        The image arrary with the segmentation labels with low priority.
    
    overlap_limit : float
        The limit of the overlap between the components
        If there is an overlap between the components lower than the limit
        the component from low priority labels is added to the high priority labels

    Returns
    -------
    np.ndarray
        The image arrary with the segmentation labels with high and low priority.
    """

    labels_union = high_priority_labels.copy()
    
    # get low priority components
    low_priority_comp = label(low_priority_labels)

    for component in np.unique(low_priority_comp[np.nonzero(low_priority_comp)]):
        
        overlap = np.mean(labels_union[low_priority_comp==component]>0)

        # If the overlap is lower than the limit, add the component to the high priority labels
        if overlap < overlap_limit:
            low_id = low_priority_labels[(low_priority_comp==component)]
            
            low_id = np.unique(low_id[np.nonzero(low_id)])[0]
            
            labels_union[low_priority_comp==component] = low_id

        
        else:
            # if has overlap, keep the high priority labels
            pass
            
            
    return labels_union



def get_new_segmentation_sample(ground_truth_map:np.ndarray, 
                                old_pred_map:np.ndarray, 
                                new_pred_map:np.ndarray, 
                                new_prob_map:np.ndarray, 
                                new_depth_map:np.ndarray,
                                data_path:str)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get the new segmentation sample based on the segmentation from the last iteration and the new segmentation prediction set
    
    Parameters
    ----------
    old_pred_map : np.ndarray
        Segmentation map from the last iteration with tree two labels
    new_pred_map : np.ndarray
        New segmentation map with tree type labels
    new_prob_map : np.ndarray
        New segmentation map with confidence/probability at each pixel
    new_depth_map : np.ndarray
        New depth map predicted by the auxiliar task of the model
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
    
    # Select only the components with confidence higher than 0.90
    new_pred_map = np.where(new_prob_map > 0.90, new_pred_map, 0)

    old_components_pred_map = label(old_pred_map)

    new_components_pred_map = label(new_pred_map)
    
    filter_components_by_mask(data_path, new_components_pred_map, new_pred_map)
    
    new_pred_map = np.where(new_depth_map > 0.3, new_pred_map, 0 )

    # remove inplace the components which is lower than 100
    filter_components_by_geometric_property(new_pred_map, 
                                            low_limit = 200, 
                                            high_limit = np.inf, 
                                            property = "area")


    new_components_pred_map = label(new_pred_map)

    # new components
    delta_label_map = get_labels_delta(old_components_img = old_components_pred_map, 
                                       new_components_img = new_components_pred_map,
                                       new_label_img = new_pred_map)
    
    # rever essa alteração
    unbalanced_delta = delta_label_map.copy()


    select_n_labels_by_class(
        delta_label_map,
        samples_by_class = 5
    )

    intersection_label_map = get_label_intersection(old_label_img = old_pred_map, 
                                                    new_label_img = new_pred_map)
    

    selected_labels_set = join_labels_set(intersection_label_map, old_pred_map, 0.10 )

    selected_labels_set = join_labels_set(delta_label_map, intersection_label_map, 0.10 )

    selected_labels_set = join_labels_set(ground_truth_map, selected_labels_set, 0.01 )

    # rever essa lógica
    all_labels_set = join_labels_set(unbalanced_delta, old_pred_map, 0.10)

    all_labels_set = join_labels_set(ground_truth_map, all_labels_set, 0.01)


    return all_labels_set, selected_labels_set



if __name__ == "__main__":
    args = read_yaml("args.yaml")

    gt_map = read_tiff("/home/luiz/multi-task-fcn/Data/orthoimages/fixed_ortoA1_25tiff.tif")

    old_pred_map = read_tiff("/home/luiz/multi-task-fcn/6.0_version_data/segmentation/samples_A1_train2tif.tif")

    new_pred_map = read_tiff("/home/luiz/multi-task-fcn/6.0_version_data/iter_001/raster_prediction/join_class_itcFalse_1.1.TIF")

    new_prob_map = read_tiff("/home/luiz/multi-task-fcn/6.0_version_data/iter_001/raster_prediction/join_prob_itcFalse_1.1.TIF")

    depth_predicted = read_tiff("/home/luiz/multi-task-fcn/6.0_version_data/iter_001/raster_prediction/depth_itcFalse_1.1.TIF")
    
    all_labels_set, selected_labels_set =  get_new_segmentation_sample(old_pred_map = old_pred_map,
                                                                       new_pred_map = new_pred_map,
                                                                       new_prob_map = new_prob_map,
                                                                       new_depth_map = depth_predicted,
                                                                       ground_truth_map = gt_map,
                                                                       data_path =  args.data_path
                                                                       )
    
    print("Ok")
    # old_pred_map = np.load("debug_arrays/old_pred_map.npy")
    # new_pred = np.load("debug_arrays/new_pred.npy")
    # ground_truth_map = np.load("debug_arrays/ground_truth_map.npy")
    # new_prob_map = np.load("debug_arrays/new_prob_map.npy")


    # all_labels_set, selected_labels_set = get_new_segmentation_sample(ground_truth_map, old_pred_map, new_pred, new_prob_map, args.data_path)




    # get_new_segmentation_sample
    # # data parameters
    # itc = False
    # sum_overlap = 1.1

    # # paths to the files from one iteration
    # current_iter_folder = "/home/luiz/multi-task-fcn/MyData/iter_001"
    # depth_path = os.path.join(
    #     current_iter_folder, "raster_prediction", f"depth_itc{itc}_{sum_overlap}.TIF"
    # )
    # prob_path = os.path.join(
    #     current_iter_folder,
    #     "raster_prediction",
    #     f"join_prob_itc{itc}_{sum_overlap}.TIF",
    # )
    # pred_path = os.path.join(
    #     current_iter_folder,
    #     "raster_prediction",
    #     f"join_class_itc{itc}_{sum_overlap}.TIF",
    # )

    # depth = read_tiff(depth_path)
    # prob = read_tiff(prob_path)
    # pred = read_tiff(pred_path)

    # ground_truth_path = os.path.join(args.data_path, args.train_segmentation_path)
    # ground_truth_img = read_tiff(ground_truth_path)

    # # at the first iteration old prediction is the same of ground truth segmentation
    # old_pred_map = ground_truth_img.copy()
    # all_labels, new_labels =  get_new_segmentation_sample(
    #     ground_truth_map=ground_truth_img, 
    #     old_pred_map = old_pred_map,
    #     new_pred_map = pred,
    #     new_prob_map = prob, 
    #     data_path=args.data_path)
    
    # print("All labels", np.unique(all_labels))
    # print("Selected labels", np.unique(new_labels))
    # print("Ground Truth labels", np.unique(ground_truth_img))

    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # ax[0].imshow(all_labels)
    # ax[1].imshow(new_labels)
    # ax[2].imshow(ground_truth_img)
    # # set title
    # ax[0].set_title("All labels")
    # ax[1].set_title("Selected labels")
    # ax[2].set_title("ground truth")

    # # # save figure
    # plt.savefig("debug_images/sample_selection_test.png", dpi=600)


