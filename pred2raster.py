"""Save predictions to raster"""
#%%
import os
from osgeo import gdal
import numpy as np
from src.utils import read_tiff
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import array2raster, read_yaml
import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_dir', default='./exp_deeplab_v4',
#                     help="Experiment directory")
# # parser.add_argument('--ref_path',type=str, default='D:/Projects/PUC-PoC/ITC_level_exp/data/imgCLASStestMATLAB.tif', 
# #                     help="Path containing the refrence and mask data for training")
# # parser.add_argument('--ref_path',type=str, default='/home/luiz/multi-task-fcn/Data/mask.tif',
#                     # help="Path containing the refrence and mask data for training")

# parser.add_argument('--ref_path',type=str, default='/home/luiz/multi-task-fcn/Data/samples_A1_train2tif.tif',
#                     help="Path containing the refrence and mask data for training")
# parser.add_argument("--overlap", type=float, default=[0.1,0.4,0.6], 
#                     help="samples per epoch")
# parser.add_argument('--train_protocol',type=int, default=1, 
#                     help="1: Random selection, 2: 4 region, 3: chessboard")
# parser.add_argument("--small_train_set", type=bool, default=False, 
#                     help="True for training with 1 fold and testing with 3. False otherwise")
# parser.add_argument("--test_itc", type=bool, default=False, 
#                     help="True for predicting only the test ITCs")
#%%

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    # args = parser.parse_args()
    args = read_yaml("args.yaml")

    plt.rcParams.update({'font.size': 10})
    sns.set_style("darkgrid")
    
    raster_src = gdal.Open(os.path.join(args.ref_path_pred2raster))
    
        
    raster_test = read_tiff(args.ref_path_pred2raster)

  
    prediction_file = os.path.join(args.model_dir,'prediction','join_class_itc{}_{}.TIF'
                              .format(args.test_itc,np.sum(args.overlap)))
    
    prob_file = os.path.join(args.model_dir,'prediction','join_prob_itc{}_{}.TIF'
                              .format(args.test_itc,np.sum(args.overlap)))
    
    
    depth_file = os.path.join(args.model_dir,'prediction','depth_itc{}_{}.TIF'
                              .format(args.test_itc,np.sum(args.overlap)))
               
    
    if not os.path.isfile(prediction_file):
    
        for ov in args.overlap:
            try:
                prediction_test = np.add(prediction_test,np.load(os.path.join(args.model_dir,
                                                                              'prediction',
                                                                              'prob_map_itc{}_{}.npy'
                                                                              .format(args.test_itc,ov))))
                
                depth_test = np.add(depth_test,np.load(os.path.join(args.model_dir,
                                                                              'prediction',
                                                                              'depth_map_itc{}_{}.npy'
                                                                              .format(args.test_itc,ov))))
                
            except:
                prediction_test = np.load(os.path.join(args.model_dir,
                                                       'prediction','prob_map_itc{}_{}.npy'
                                                       .format(args.test_itc,ov)))
                
                depth_test = np.load(os.path.join(args.model_dir,
                                                       'prediction','depth_map_itc{}_{}.npy'
                                                       .format(args.test_itc,ov)))
                
        
        prediction_test/=3
        depth_test/=3

        
        array2raster(prediction_file, raster_src, np.argmax(prediction_test,axis=-1), "Byte")    
        array2raster(prob_file, raster_src, np.amax(prediction_test,axis=-1), "Float32")  
        array2raster(depth_file, raster_src, depth_test, "Float32")   
        
 