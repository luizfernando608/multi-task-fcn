import sys
from typing import Literal, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from os.path import dirname, join

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.io_operations import get_image_metadata, get_image_shape, read_tiff, check_file_extension, read_window_around_coord, convert_tiff_to_npy, get_npy_filepath_from_tiff, get_npy_shape, load_npy_memmap



def oversample(coords: np.ndarray, 
               coords_label: np.ndarray, 
               method: Literal["max", "median"]) -> np.ndarray:
    """Oversamples data to balance classes based on segmentation samples.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the segmentation samples where non-zero values exist.
    coords_label : np.ndarray
        Segmentation labels corresponding to non-zero values. Each value relates to the pixel label at the `coords` position.
    method : Literal["max", "median"]
        The oversampling method to balance the tree-type classes.
        - "max" for maximizing class counts.
        - "median" for achieving class counts closer to the median.

    Returns
    -------
    np.ndarray
        Balanced segmentation sample coordinates with non-zero values.
        The coordinates are adjusted to balance tree-type classes based on the chosen oversampling method.
    """

    
    uniq, count = np.unique(coords_label, return_counts=True)
    
    if method == "max":
        upper_samp_limit = np.max(count)
    
    elif method == "median":
        upper_samp_limit = int(np.median(count))
    
    else:
        raise NotImplementedError(f"Method ``{method}`` were not implemented yet.")

    out_coords = np.zeros( (upper_samp_limit*len(uniq), 2), dtype='int64')
    

    for j in range(len(uniq)):

        lab_ind = np.where(coords_label == uniq[j]) 

        # If num of samples where the class is present is less than max_samp
        # then we need to oversample
        if len(lab_ind[0]) < upper_samp_limit:
            # Randomly select samples with replacement to match max_samp
            index = np.random.choice(lab_ind[0], upper_samp_limit, replace=True)
            # Add to output array
            out_coords[j*upper_samp_limit:(j+1)*upper_samp_limit,:] = coords[index]
            
        # If the number of samples where the class is present is the same as max_samp
        # then we don't need to oversample, just add the samples randomly to the output array
        else:
            # Randomly select samples without replacement
            index = np.random.choice(lab_ind[0], upper_samp_limit, replace=False)
            # Add to output array
            out_coords[j*upper_samp_limit:(j+1)*upper_samp_limit,:] = coords[index]
    
    # shuffle out_coords order
    np.random.shuffle(out_coords)

    return out_coords




class dataset_with_lazy_loading_window(Dataset):
    def __init__(self,
                 image_path:str,
                 segmentation_path:str,
                 distance_map_path:str,
                 crop_size:int,
                 dataset_type:Literal["train", "val", "test"],
                 samples:int = None,
                 augment:bool = False,
                 overlap_rate:float = None
                 ) -> None:
        
        super().__init__()

        if (dataset_type == "test") and (overlap_rate == None):
            raise ValueError("Provide 'overlap_rate' for 'test' dataset_type")

        if (dataset_type == "test") and (augment):
            raise ValueError("'test' dataset_type does not accept augmentation")
        
        if overlap_rate is not None:
            if (overlap_rate >= 1) or (overlap_rate <= 0):
                raise ValueError("overlap_rate only accept values between (0.0, 1.0) ")

        if segmentation_path is not None:
            check_file_extension(segmentation_path, ".npy")
        
        if image_path is not None:
            check_file_extension(image_path, ".npy")
        
        if distance_map_path is not None:
            check_file_extension(distance_map_path, ".npy")
        
        
        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.distance_map_path = distance_map_path
        
        self.samples = samples
        self.crop_size = crop_size
        self.dataset_type = dataset_type
        self.augment = augment

        self.overlap_rate = overlap_rate
        
        if dataset_type in ["train", "val"]:
            self.img_segmentation = load_npy_memmap(segmentation_path)
        
        self.image_shape = get_npy_shape(image_path)

        self.generate_coords()



    
    def generate_coords(self):

        if self.dataset_type == "train":
            
            coords = np.where(self.img_segmentation!=0)
            coords = np.array(coords)
            coords = np.rollaxis(coords, 1, 0)
            
            coords_label = self.img_segmentation[np.nonzero(self.img_segmentation)]

            coords = oversample(coords, coords_label, "median")


        elif self.dataset_type == "val":

            coords = np.where(self.img_segmentation!=0)
            coords = np.array(coords)
            coords = np.rollaxis(coords, 1, 0)
            
            coords_label = self.img_segmentation[np.nonzero(self.img_segmentation)]

            coords = oversample(coords, coords_label, "median")

        
        elif self.dataset_type == "test":            
            coords_list = []

            self.overlap_size = int(self.crop_size * self.overlap_rate)
            self.stride_size = self.crop_size - self.overlap_size

            for m in range(self.stride_size//2, self.image_shape[1] + self.stride_size//2, self.stride_size):

                for n in range(self.stride_size//2, self.image_shape[2] + self.stride_size//2, self.stride_size):
    
                    coords_list.append([m,n])
            
            coords = np.array(coords_list)

        self.coords = np.array(coords)





    def __len__(self):

        if (self.samples is None):
            return len(self.coords)

        if (self.samples > len(self.coords)):
            return len(self.coords)
        
        
        return self.samples
        
        
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the data from the dataset
        
        Parameters
        ----------
        idx : int
            The index of the data to be loaded
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The image crop, depth map crop, and label ref crop
        
        """
        current_coord = self.coords[idx].copy()
        
        
        if self.augment:
            
            # Run random shift
            uniform_dist_range = (-0.9, 0.9)
            
            random_row_prop = np.random.uniform(*uniform_dist_range)
            random_column_prop = np.random.uniform(*uniform_dist_range)

            current_coord[0] += int(random_row_prop * (self.crop_size//2))
            current_coord[1] += int(random_column_prop * (self.crop_size//2))


        image = read_window_around_coord(
            coord=current_coord, 
            image_npy_path=self.image_path,
            crop_size=self.crop_size
        )
        
        # normalize by 255
        image = (image/255)

        if self.dataset_type == "test":
            
            return image, current_coord

        segmentation = read_window_around_coord(
            coord=current_coord, 
            image_npy_path=self.segmentation_path,
            crop_size=self.crop_size
        )

        distance_map = read_window_around_coord(
            coord=current_coord, 
            image_npy_path=self.distance_map_path,
            crop_size=self.crop_size
        )


        if self.augment:
            # Run Horizontal Flip
            if np.random.random() > 0.5:
                image = transforms.functional.hflip(image)
                segmentation = transforms.functional.hflip(segmentation)
                distance_map = transforms.functional.hflip(distance_map)

            # Run Vertical Flip
            if np.random.random() > 0.5:
                image = transforms.functional.vflip(image)
                segmentation = transforms.functional.vflip(segmentation)
                distance_map = transforms.functional.vflip(distance_map)
            
            # Run random rotation
            angle = int(np.random.choice([0, 90, 180, 270]))
            
            image = transforms.functional.rotate(image.unsqueeze(0), angle).squeeze(0)
            segmentation = transforms.functional.rotate(segmentation.unsqueeze(0), angle).squeeze(0)
            distance_map = transforms.functional.rotate(distance_map.unsqueeze(0), angle).squeeze(0)


        return image, distance_map, segmentation.long()            


if __name__ == "__main__":
    
    from torch.utils.data import DataLoader

    orthoimage_path = join(ROOT_PATH,
    "amazon_mc_input_data/orthoimage/NOV_2017_FINAL_004.tif")
    
    segmentation_path = join(ROOT_PATH,"amazon_mc_input_data/segmentation/train_set.tif")

    distance_path = join(ROOT_PATH,"0.0_test_data/iter_000/distance_map/train_distance_map.tif")

    orthoimage_npy_path = get_npy_filepath_from_tiff(orthoimage_path)

    segmentation_npy_path = get_npy_filepath_from_tiff(segmentation_path)

    distance_npy_path = get_npy_filepath_from_tiff(
        distance_path
    )
    
    convert_tiff_to_npy(orthoimage_path)
    convert_tiff_to_npy(segmentation_path)
    convert_tiff_to_npy(distance_path)

    dataset = dataset_with_lazy_loading_window(
        image_path=orthoimage_npy_path,
        segmentation_path=segmentation_npy_path,
        distance_map_path=distance_npy_path,
        augment=True,
        dataset_type="train",
        samples=2000,
        crop_size=128
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 4,
        num_workers = 0,
        pin_memory = True,
        drop_last = True,
        shuffle = True,
    )
    
    for data in train_loader:
        
        print(type(data))




    pass
