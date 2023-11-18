import os
from os.path import dirname, join
import numpy as np

from skimage.measure import label, regionprops_table

from src.utils import read_tiff, read_yaml, fix_relative_paths

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

from imblearn.under_sampling import RandomUnderSampler

from typing import List, Tuple

from generate_distance_map import apply_gaussian_distance_map
from scipy.ndimage import gaussian_filter

from src.metrics import evaluate_component_metrics


args = read_yaml(join(dirname(__file__), "args.yaml"))
fix_relative_paths(args)

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


def set_same_class_at_component(label_img:np.ndarray):

    components_img = label(np.where(label_img>0, 1, 0))
    
    # remove component 0
    component_ids = np.unique(components_img)
    component_ids = component_ids[np.nonzero(component_ids)]

    for component_id in tqdm(component_ids):
        filter_component = components_img == component_id

        ids, counts = np.unique(label_img[filter_component], return_counts=True)

        most_commom_id = ids[np.argmax(counts)]

        label_img[filter_component] = most_commom_id
    


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
    new_components_img = label(new_label_img)

    old_components_img = label(old_label_img)

    label_delta = np.zeros_like(new_components_img)

    components_to_iter = np.unique(new_components_img)
    components_to_iter = components_to_iter[np.nonzero(components_to_iter)]

    print("Selecting the new components that are not in old segmentation")
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

    print("Getting the intersection between the old and new segmentation")
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
    
    # random shufle dataframe
    delta_stats = delta_stats.sample(frac=1, random_state=0)
    
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




def filter_components_by_mask(pred_map:np.ndarray):
    """Remove labels and components out of the mask.tif area

    Parameters
    ----------
    data_path : str
        Path to the data folder
    pred_map : np.ndarray
        The labels map from the current iteration.
    """
    
    mask = read_tiff(args["mask_path"])
    mask = np.where(mask == 99, False, True)

    components_pred_map = label(pred_map)
    
    # Components with some peace out the mask
    components_to_check = np.unique(components_pred_map[mask])
    components_to_check = components_to_check[np.nonzero(components_to_check)]
    
    print("Filtering components out of the area of the experiment")
    for component in tqdm(np.unique(components_to_check)):

        component_filter = components_pred_map==component
        
        area_out_mask = np.mean( mask[component_filter])
        
        # if more than 20% of the component is out, remove it
        if area_out_mask >= 0.20:
            pred_map[component_filter] = 0
            components_pred_map[component_filter] = 0

    # cut out anything out of the area
    pred_map[mask] = 0


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


def select_good_samples(old_pred_map:np.ndarray,
                        new_pred_map:np.ndarray, 
                        new_prob_map:np.ndarray, 
                        new_depth_map:np.ndarray,
                        ) -> np.ndarray:
    """
    This function defines the rules to select good samples based on model outputs.

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
    np.ndarray
        new_pred_map with the selected samples
    """

    new_pred_map = new_pred_map.copy()
    
    # depth map
    depth_gauss = gaussian_filter(new_depth_map, sigma = 9)

    # prob map
    prob_gauss = gaussian_filter(new_prob_map, sigma = 9)

    new_pred_map = np.where((depth_gauss > 0.2) & (prob_gauss > 0.9), new_pred_map, 0)

    # filter components too small or too large
    filter_components_by_geometric_property(new_pred_map, 
                                            low_limit = 500, 
                                            high_limit = np.inf, # high limit area
                                            property = "area")
    
    filter_components_by_mask(new_pred_map)
    
    # Calculate main metrics of each tree
    comp_old_pred = label(old_pred_map)
    comp_old_stats = get_components_stats(comp_old_pred, old_pred_map).reset_index()
    comp_old_stats = comp_old_stats.groupby("tree_type")[["extent", "solidity", "eccentricity", "area"]].median()
    comp_old_stats.columns  = "ref_" + comp_old_stats.columns 

    # Get metrics about the new labels
    comp_new_pred = label(new_pred_map)
    comp_new_stats =  get_components_stats(comp_new_pred, new_pred_map).reset_index()
    # Join data from the last with the new one
    comp_new_stats = comp_new_stats.merge(comp_old_stats, on = "tree_type", how = "left")
    
    comp_new_stats["dist_area"] =  np.abs(comp_new_stats["area"] - comp_new_stats["ref_area"])/comp_new_stats["ref_area"]

    comp_new_stats["diff_soli"] =  (comp_new_stats["solidity"] - comp_new_stats["ref_solidity"])
    # Select componentes based on some metrics
    selected_comp = comp_new_stats[(comp_new_stats["area"] > 800) # higher than 500
                                   & (comp_new_stats["dist_area"] < 0.1) # area between 10% less or higher
                                   & (comp_new_stats["diff_soli"] >= -0.05) # solidity
                                   ].copy()

    new_pred_map =  np.where(np.isin(comp_new_pred, selected_comp["label"].unique()), new_pred_map, 0)

    return new_pred_map



def get_new_segmentation_sample(ground_truth_map:np.ndarray, 
                                old_selected_labels:np.ndarray,
                                old_all_labels:np.ndarray,
                                new_pred_map:np.ndarray, 
                                new_prob_map:np.ndarray, 
                                new_depth_map:np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        New segmentation map with tree type labels with all components with confidence higher than 0.90

    selected_labels_set : np.array
        The same set as all_labels_set but with some filters applied to minimize unbalanced classes problem
    """
    
    # set labels at the same scale as ground truth labels
    new_pred_map += 1

    new_pred_map = select_good_samples(
        old_all_labels,
        new_pred_map,
        new_prob_map,
        new_depth_map
    )

    # Join all labels set with new prediction
    new_labels_set = join_labels_set(new_pred_map, old_all_labels)

    # Select components that are in new but not in old
    delta_label_map = get_labels_delta(old_label_img = old_selected_labels, 
                                       new_label_img = new_labels_set)
    

    unbalanced_delta = delta_label_map.copy()


    select_n_labels_by_class(
        delta_label_map,
        samples_by_class = 5
    )

    # get the new predicted shapes for components from old segmentation
    intersection_label_map = get_label_intersection(old_label_img = old_selected_labels, 
                                                    new_label_img = new_labels_set)
        
    # join updated shapes with the old ones that were not updated
    old_selected_labels_updated = join_labels_set(intersection_label_map, old_selected_labels, 0.10 )


    # join the old labels set with the new labels. balanced sample addition
    selected_labels_set = join_labels_set(delta_label_map, old_selected_labels_updated, 0.10 )
    
    # Adding the ground truth segmentation
    selected_labels_set = join_labels_set(ground_truth_map, selected_labels_set, 0.01 )



    # join the old labels set with the new labels. unbalanced sample addition
    all_labels_set = join_labels_set(unbalanced_delta, old_selected_labels_updated, 0.10)

    # Adding the ground truth segmentation
    all_labels_set = join_labels_set(ground_truth_map, all_labels_set, 0.01)


    return all_labels_set, selected_labels_set



if __name__ == "__main__":
    args = read_yaml("args.yaml")
    ROOT_PATH = dirname(__file__)
    
    version_folder = join(ROOT_PATH, "11.0_version_data")
    
    gt_map = read_tiff(f"{version_folder}/segmentation/samples_A1_train2tif.tif")

    test_gt_map = read_tiff(f"{version_folder}/segmentation/samples_A1_train2tif.tif")
    
    old_all_labels = read_tiff(f"{version_folder}/iter_001/new_labels/all_labels_set.tif")

    old_selected_labels = read_tiff(f"{version_folder}/iter_001/new_labels/selected_labels_set.tif")
                               
    new_pred_map = read_tiff(f"{version_folder}/iter_002/raster_prediction/join_class_itcFalse_1.1.TIF")

    new_prob_map = read_tiff(f"{version_folder}/iter_002/raster_prediction/join_prob_itcFalse_1.1.TIF")

    depth_predicted = read_tiff(f"{version_folder}/iter_001/raster_prediction/depth_itcFalse_1.1.TIF")
    
    all_labels_set, selected_labels_set =  get_new_segmentation_sample(old_selected_labels = old_selected_labels,
                                                                       old_all_labels = old_all_labels,

                                                                       new_pred_map = new_pred_map,
                                                                       new_prob_map = new_prob_map,
                                                                       new_depth_map = depth_predicted,
                                                                       
                                                                       ground_truth_map = gt_map
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


