import os
import matplotlib
matplotlib.use('MacOSX')

import tqdm

from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)

from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

import shutil



def rgb_reconstruction():

    """
    Reconstruct scene using RGB images only taken from 2 RGBD cameras
    Try with and without known camera poses.

    based on:
    https://nbviewer.org/github/cvg/Hierarchical-Localization/blob/master/demo.ipynb
    https://nbviewer.org/github/cvg/Hierarchical-Localization/blob/master/pipeline_SfM.ipynb
    """

    images = Path("/Users/shilem2/OneDrive - Medtronic PLC/projects/rgbd_reconstruction/data/work_volume_data/20240327_091906_sample/")

    outputs = Path("outputs/work_volume/")
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    # retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superglue"]

    # 3D mapping
    references = [p.relative_to(images).as_posix() for p in (images / 'images').iterdir()]
    print(len(references), "mapping images")
    image_list = [read_image(images / r) for r in references]
    plot_images(image_list, dpi=25)

    # extract features and match them across image pairs
    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # run incremental Structure-From-Motion
    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True)
    fig.show()

    visualization.visualize_sfm_2d(model, images, color_by="visibility", n=2)


    pass


if __name__ == '__main__':

    rgb_reconstruction()

    pass