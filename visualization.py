
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.measure import find_contours

from src.utils import check_folder, read_tiff, run_in_thread, run_in_process

@run_in_process
def generate_labels_view(current_iter_folder:str, orthoimage_path:str):
    """Function to generate images for qualitative evaluation.
    These images are not used for any kind of numeric evaluation.

    Parameters
    ----------
    current_iter_folder : str
        Path to iteration folder
    
    orthoimage_path : str
        Path to remote sensing orthoimage
    """

    
    ALL_LABELS_PATH = join(current_iter_folder, "new_labels", "all_labels_set.tif")
    SELECTED_LABELS_PATH = join(current_iter_folder, "new_labels", "selected_labels_set.tif")
    ORTHO_IMAGE_PATH = orthoimage_path

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

    # Plot all labels contour
    CONTOUR_ALL_LABELS_OUT_FOLDER = join(OUTPUT_MAP_FOLDER, "all_labels_contour")
    check_folder(CONTOUR_ALL_LABELS_OUT_FOLDER)

    plt.figure(dpi = 1200)

    ORTHOIMAGE = read_tiff(ORTHO_IMAGE_PATH)
    plt.imshow(np.moveaxis(ORTHOIMAGE, 0, 2))
    del ORTHOIMAGE

    # plot contours
    for contour in find_contours(ALL_LABELS_MAP):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=0.1, color = "red", label = "Predição")

    plt.axis('off')
    plt.savefig(join(CONTOUR_ALL_LABELS_OUT_FOLDER, f"{current_iter:03d}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    

if __name__ == "__main__":
    
    VERSION_FOLDER = "2.7_version_data"
    ORTHOIMAGE_PATH = join(dirname(__file__), "amazon_md_input_data", "orthoimage", "NOV_2017_FINAL_004.tif")
    ITER_NUM = 2
    
    DATA_PATH = join(dirname(__file__), VERSION_FOLDER)
    
    

    current_iter_folder = join(DATA_PATH, f"iter_{ITER_NUM:03d}")

    
    generate_labels_view(current_iter_folder, ORTHOIMAGE_PATH)
    