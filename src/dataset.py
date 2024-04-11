import sys
from typing import Literal,Tuple

import numpy as np
import torch

from os.path import dirname, join
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_PATH = dirname(dirname(__file__))
sys.path.append(ROOT_PATH)

from src.utils import get_crop_image, get_pad_width, oversample
from src.io_operations import check_file_extension, get_npy_shape



class DatasetFromCoord(Dataset):
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
            self.img_segmentation = np.load(segmentation_path)
            self.img_depth = np.load(distance_map_path)
        
        self.image = np.load(image_path)
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


    def read_window_around_coord(self, coord:np.ndarray, image:np.ndarray) -> torch.Tensor:
        
        image_crop = get_crop_image(image, image.shape, coord, self.crop_size)

        pad_width = get_pad_width(self.crop_size, coord, image.shape)

        # apply padding to image
        image_crop = np.pad(
            image_crop, 
            pad_width = pad_width,
            mode = "constant",
            constant_values = 0
        )
        

        if (image_crop.shape[-1] != self.crop_size) or (image_crop.shape[-2] != self.crop_size):
            raise ValueError(f"There is a bug relationed to the shape {image_crop.shape}")

        return torch.tensor(image_crop)


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


        image = self.read_window_around_coord(
            coord=current_coord,
            image=self.image,
        )
        
        # normalize by 255
        image = (image/255)

        if self.dataset_type == "test":
            
            return image, current_coord

        segmentation = self.read_window_around_coord(
            coord=current_coord,
            image=self.img_segmentation
        )

        distance_map = self.read_window_around_coord(
            coord=current_coord,
            image=self.img_depth
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

    
    def __len__(self):

        if (self.samples is None):
            return len(self.coords)

        if (self.samples > len(self.coords)):
            return len(self.coords)
        
        
        return self.samples
    
