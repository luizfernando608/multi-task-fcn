import os
from os.path import dirname, join
from typing import Literal, Tuple

import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import read_tiff, read_yaml

ROOT_PATH = dirname(dirname(__file__))
ARGS = read_yaml(join(ROOT_PATH, "args.yaml"))

class DatasetFromCoord(Dataset):
    """Dataset to load training data"""
    def __init__(self, 
                image:np.ndarray = None,
                labels:np.ndarray = None,
                depth_img:np.ndarray = None,
                coords:np.ndarray = None, 
                psize: int = None,
                samples:int = False,
                evaluation:bool = False,
                augment:bool = False):
        """Initialize the dataset to load training data
        It defines the augmentation methods and the data to be loaded in the model
        This instance folows the pytorch Dataset pattern

        Parameters
        ----------
        image : np.ndarray, optional
            The real image from remote sensing, by default None
        labels : np.ndarray, optional
            The segmentation of trees in image, by default None
        depth_img : np.ndarray, optional
            The segmentation with filters like distance map and gaussian blur, by default None
        coords : np.ndarray, optional
            The positions in the ground_truth segmentation where the values are different of 0, by default None
        psize : int, optional
            The side of the square to be cut from the image.
            This box involves the position of the tree and the border of the box.
            The box is centered in the tree position.
            by default ``None``
        samples : int, optional
            Num samples per epoch, by default False
        evaluation : bool, optional
            Define the loading method for training or evaluation.
                True: return just the image crop,
                False: return image, depth map, and label ref croppped
            by default False
        augment : bool, optional
            If True applies data augmentation for training, by default False
        """

        super(DatasetFromCoord, self).__init__()
        self.image = image
        self.labels = labels
        self.depth_img = depth_img
        self.coord = coords
        self.psize = psize
        self.samples = samples
        self.evaluation = evaluation
        
        self.augment = augment
        

    def __len__(self):
        """Get the Dataset length

        Returns
        -------
        int
            The number of samples in the dataset
        """
        
        if self.samples:
            return self.samples
        
        
        else:
            return len(self.coord)
        
    

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
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        if self.evaluation:
            # Crop image around the coords with a box with side psize
            image = self.image[:, 
                            self.coord[idx, 0] - self.psize//2 : self.coord[idx,0] + self.psize//2,
                            self.coord[idx,1] - self.psize//2 : self.coord[idx,1] + self.psize//2]
 

        
        # If is training, return image, depth map, and label ref croppped
        if not self.evaluation:

            # If augmentation is enabled, random crop the label around the position of the coord
            if self.augment:
                # random distance between the label coord and the left side bound box
                random_x_prop = np.random.uniform(low = 0.1, high = 0.9)
                # the distance between the label center and the left bound box
                dist_x_left_border = int(self.psize * random_x_prop)

                # random distance between the label coord and the top side bound box
                random_y_prop = np.random.uniform(low = 0.1, high = 0.9)
                # the distance between the label center and the right bound box
                dist_y_top_border = int(self.psize * random_y_prop)


            else:
                dist_x_left_border = self.psize//2

                dist_y_top_border = self.psize//2
            

            dist_x_right_border = self.psize - dist_x_left_border
            dist_y_bottom_border = self.psize - dist_y_top_border

            # Crop image around the coords with a box with side psize
            image = self.image[:, 
                            self.coord[idx, 0] - dist_x_left_border: self.coord[idx,0] + dist_x_right_border,
                            self.coord[idx, 1] - dist_y_bottom_border : self.coord[idx,1] + dist_y_top_border]
            
            # crop the label ref with the same box
            ref = self.labels[self.coord[idx, 0] - dist_x_left_border: self.coord[idx,0] + dist_x_right_border,
                              self.coord[idx,1] - dist_y_bottom_border : self.coord[idx,1] + dist_y_top_border]
            
            
            # crop the depth map with the same box
            depth = self.depth_img[self.coord[idx, 0] - dist_x_left_border: self.coord[idx,0] + dist_x_right_border,
                                   self.coord[idx,1] - dist_y_bottom_border : self.coord[idx,1] + dist_y_top_border]

            

            image = torch.from_numpy(image.astype(np.float32))

            ref = torch.tensor(ref.astype(np.float32))       
            
            depth = torch.tensor(depth.astype(np.float32)) 


            # # If true, applies data augmentation
            if self.augment:
                # Run Horizontal Flip
                if np.random.random() > 0.5:
                    image = transforms.functional.hflip(image)
                    ref = transforms.functional.hflip(ref)
                    depth = transforms.functional.hflip(depth)

                # Run Vertical Flip
                if np.random.random() > 0.5:
                    image = transforms.functional.vflip(image)
                    ref = transforms.functional.vflip(ref)
                    depth = transforms.functional.vflip(depth)
                
                # Run random rotation
                angle = int(np.random.choice([0, 90, 180, 270]))
                
                image = transforms.functional.rotate(image.unsqueeze(0), angle).squeeze(0)
                ref = transforms.functional.rotate(ref.unsqueeze(0), angle).squeeze(0)
                depth = transforms.functional.rotate(depth.unsqueeze(0), angle).squeeze(0)


            return image, depth, ref.long()

        # If evaluation is True, return just the image crop
        return image



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



class DataSetFromImagePath(Dataset):
    def __init__(self,
                 image_path:str,
                 segmentation_path:str,
                 distance_map_path:str,
                 samples:int,
                 crop_size:int,
                 dataset_type:Literal["train", "val", "test"],
                 augment:bool,
                 overlap_rate = None
                 ) -> None:
        
        super().__init__()

        if (dataset_type == "test") and (overlap_rate == None):
            raise ValueError("Provide 'overlap_rate' for 'test' dataset_type")

        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.distance_map_path = distance_map_path
        
        self.samples = samples
        self.crop_size = crop_size
        self.dataset_type = dataset_type
        self.augment = augment

        self.overlap_rate = overlap_rate

        self.img_segmentation = read_tiff(segmentation_path)

        self.generate_coords()



    
    def generate_coords(self):

        if self.dataset_type == "train":
            
            coords = np.where(self.img_segmentation!=0)
            coords = np.array(coords)
            coords = np.rollaxis(coords, 1, 0)
            
            coords_label = self.img_segmentation[np.nonzero(self.img_segmentation)]

            coords = oversample(coords, coords_label, "max")


        elif self.dataset_type == "val":

            coords = np.where(self.img_segmentation!=0)
            coords = np.array(coords)
            coords = np.rollaxis(coords, 1, 0)
            
            coords_label = self.img_segmentation[np.nonzero(self.img_segmentation)]

            coords = oversample(coords, coords_label, "median")

        
        elif self.dataset_type == "test":            
            coords_list = []

            overlap_size = int(self.crop_size * self.overlap_rate)
            stride_size = self.crop_size - overlap_size

            for m in range(self.crop_size//2, self.img_segmentation.shape[0], stride_size):

                for n in range(self.crop_size//2, self.img_segmentation.shape[1], stride_size):
    
                    coords_list.append([m,n])
            
            coords = np.array(coords_list)

        self.coords = np.array(coords)


    def read_window(self, coord:np.ndarray, img_path:str) -> torch.Tensor:


        pad_width = [[0,0], [0,0]]
        
        # check row
        start_row = coord[0] - self.crop_size // 2
        if start_row < 0:
            
            start_row = 0
            
            pad_width[0][0] = abs(coord[0] - self.crop_size // 2)
            


        end_row = coord[0] + self.crop_size // 2
        if end_row > self.img_segmentation.shape[0]:
            
            end_row = self.img_segmentation.shape[0]
            
            pad_width[0][1] = abs(coord[0] + self.crop_size//2 - end_row)
        

        # check column
        start_column = coord[1] - self.crop_size // 2

        if start_column < 0:
            
            start_column = 0
            
            pad_width[1][0] = abs(coord[1] - self.crop_size // 2)


        end_column = coord[1] + self.crop_size // 2
        
        if end_column > self.img_segmentation.shape[1]:
            
            end_column = self.img_segmentation.shape[1]

            pad_width[1][1] = abs(coord[1] + (self.crop_size // 2) - end_column)



        window = rasterio.windows.Window.from_slices(
            (start_row, end_row), 
            (start_column, end_column)
        )

        with rasterio.open(img_path) as src:
            image_crop = src.read(window = window)

        if len(image_crop.shape) == 3:
            pad_width = [[0,0]] + pad_width


        image_crop = np.pad(
            image_crop, 
            pad_width =  pad_width,
            mode = "constant",
            constant_values = 0
        )
        

        if (image_crop.shape[-1] != self.crop_size) or (image_crop.shape[-2] != self.crop_size):
            raise ValueError(f"There is a bug relationed to the shape {image_crop.shape}")

        return torch.tensor(image_crop)


    def __len__(self):
        
        if (self.samples > len(self.coords)) or (self.samples is None):
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
        current_coord = self.coords[idx]
        
        image = self.read_window(current_coord, self.image_path)
        segmentation = self.read_window(current_coord, self.segmentation_path)
        distance_map = self.read_window(current_coord, self.distance_map_path)

        return image, distance_map, segmentation


if __name__ == "__main__":
    from utils import fix_relative_paths

    args = read_yaml("args.yaml")
    fix_relative_paths(args=args)
    
    ORTHO_PATH = args.ortho_image
    
    SEG_PATH  = args.train_segmentation_path
    
    DIST_MP_PATH = join(
        ROOT_PATH,
        r"1.0_amazon_version_data\iter_000\distance_map\train_distance_map.tif"
    )


    # build data for training
    train_dataset = DataSetFromImagePath(
        image_path = ORTHO_PATH,
        segmentation_path = SEG_PATH,
        distance_map_path = DIST_MP_PATH,
        samples = 1000,
        crop_size = 5000,
        dataset_type = "test",
        overlap_rate = 0.3,
        augment = True
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )


    for it, (inp_img, depth, ref) in enumerate(tqdm(train_loader)):  
        print("oi")
        pass