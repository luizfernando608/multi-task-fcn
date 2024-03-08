
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.measure import find_contours

from src.utils import check_folder, read_tiff, run_in_thread, run_in_process

# generate view only for sythentic labels
def generate_view_for_sythentic_label(current_iter_folder:str, train_segmentation_path:str, orthoimage_path:str):

    current_iter = int(current_iter_folder.split("iter_")[-1])

    ALL_LABELS_PATH = join(current_iter_folder, "new_labels", "all_labels_set.tif")
    ALL_LABELS_MAP = read_tiff(ALL_LABELS_PATH)

    TRAIN_GT_MAP = read_tiff(train_segmentation_path)

    synthetic_labels = np.where(TRAIN_GT_MAP > 0, 0, ALL_LABELS_MAP)
    
    # folder to save view
    OUTPUT_MAP_FOLDER = join(dirname(current_iter_folder), "visualization", "synthetic_all_labels")
    # create folder if it doesnt exists
    check_folder(OUTPUT_MAP_FOLDER)


    plt.figure(dpi = 1200)

    ORTHOIMAGE = read_tiff(orthoimage_path)
    plt.imshow(np.moveaxis(ORTHOIMAGE, 0, 2))
    del ORTHOIMAGE

    # plot contours
    for contour in find_contours(synthetic_labels):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color = "red", label = "Predição")

    plt.axis('off')
    plt.savefig(join(OUTPUT_MAP_FOLDER, f"{current_iter:03d}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close()



def generate_labels_view(current_iter_folder:str, orthoimage_path:str, train_segmentation_path:str):
    """Function to generate images for qualitative evaluation.
    These images are not used for any kind of numeric evaluation.

    Parameters
    ----------
    current_iter_folder : str
        Path to iteration folder
    
    orthoimage_path : str
        Path to remote sensing orthoimage
    
    train_segmentation_path : str
        Path to ground_truth segmentation used for train
    """

    
    ALL_LABELS_PATH = join(current_iter_folder, "new_labels", "all_labels_set.tif")
    SELECTED_LABELS_PATH = join(current_iter_folder, "new_labels", "selected_labels_set.tif")

    current_iter = int(current_iter_folder.split("iter_")[-1])

    ALL_LABELS_MAP = read_tiff(ALL_LABELS_PATH)
    SELECTED_LABELS_MAP = read_tiff(SELECTED_LABELS_PATH)
    

    num_classes = np.unique(ALL_LABELS_MAP[ALL_LABELS_MAP != 0]).shape[0]

    if num_classes == 14:
        DEFAULT_COLORS = ('silver', 'blue', 'yellow', 'magenta', 'green', 
                        'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen', 
                        'red', 'darkgreen', 'gold', 'teal')
    
    else:
        from seaborn import color_palette
        DEFAULT_COLORS = color_palette("tab20", num_classes)
    

    OUTPUT_MAP_FOLDER = join(dirname(current_iter_folder), "visualization")
    
    # create output folder
    check_folder(OUTPUT_MAP_FOLDER)

    # PLOT ALL LABELS
    ALL_LABELS_OUT_FOLDER = join(OUTPUT_MAP_FOLDER, "all_labels",)
    check_folder(ALL_LABELS_OUT_FOLDER)
    
    plt.figure(dpi = 300)
    plt.imshow(label2rgb(ALL_LABELS_MAP, colors = DEFAULT_COLORS))
    plt.axis('off')
    plt.savefig(join(ALL_LABELS_OUT_FOLDER, f"{current_iter:03d}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close()


    # PLOT SELECTED LABELS
    SELECTED_LABELS_OUT_FOLDER = join(OUTPUT_MAP_FOLDER, "selected_labels")
    check_folder(SELECTED_LABELS_OUT_FOLDER)

    plt.figure(dpi = 300)
    plt.imshow(label2rgb(SELECTED_LABELS_MAP, colors = DEFAULT_COLORS))
    del SELECTED_LABELS_MAP
    plt.axis('off')
    plt.savefig(join(SELECTED_LABELS_OUT_FOLDER, f"{current_iter:03d}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close()


    generate_view_for_sythentic_label(
        current_iter_folder=current_iter_folder,
        train_segmentation_path=train_segmentation_path,
        orthoimage_path=orthoimage_path
    )


    

if __name__ == "__main__":
    from src.utils import load_args
    
    # parameters
    VERSION_FOLDER = "2.10_version_data"
    ITER_NUM = 1


    args = load_args(join(dirname(__file__), VERSION_FOLDER, "args.yaml"))
    current_iter_folder = join(args.data_path, f"iter_{ITER_NUM:03d}")


    generate_labels_view(
        current_iter_folder = current_iter_folder,
        orthoimage_path = args.ortho_image,
        train_segmentation_path = args.train_segmentation_path
    )