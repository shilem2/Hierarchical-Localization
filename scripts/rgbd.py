import os
import matplotlib
matplotlib.use('MacOSX')

import tqdm

from pathlib import Path
import numpy as np

import pycolmap

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)

from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

from hloc.pipelines.Aachen_v1_1.pipeline_loftr import logger, match_dense, pairs_from_covisibility, triangulation, \
    pairs_from_retrieval, localize_sfm, pformat


import shutil



def rgb_reconstruction_based_on_demo():

    """
    Reconstruct scene using RGB images only taken from 2 RGBD cameras
    Try with and without known camera poses.

    based on:
    https://nbviewer.org/github/cvg/Hierarchical-Localization/blob/master/demo.ipynb
    """

    images = Path("/Users/shilem2/OneDrive - Medtronic PLC/projects/rgbd_reconstruction/data/work_volume_data/20240327_091906_sample/")

    outputs = Path("../outputs/work_volume/based_on_demo/")
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    # retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superglue"]

    shutil.rmtree(outputs.as_posix(), ignore_errors=True)

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


def rgb_reconstruction_colmap_dense():
    """
    Not working due to CUDA compilation requirerment!

    Dense rconstruction using pycolmap directly

    based on:
    https://github.com/colmap/colmap/tree/main/pycolmap#reconstruction-pipeline
    """

    image_dir = Path("/Users/shilem2/OneDrive - Medtronic PLC/projects/rgbd_reconstruction/data/work_volume_data/20240327_091906_sample/images")
    output_path = Path("../outputs/work_volume/colmap_dense")

    shutil.rmtree(output_path.as_posix(), ignore_errors=True)

    output_path.mkdir(parents=True, exist_ok=True)
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

    pass

def rgb_reconstruction_semi_dense_pipeline_loftr():

    # images = Path("/Users/shilem2/OneDrive - Medtronic PLC/projects/rgbd_reconstruction/data/work_volume_data/20240327_091906_sample/")

    # outputs = Path("../outputs/work_volume/based_on_loftr/")
    # shutil.rmtree(outputs.as_posix(), ignore_errors=True)
    # sfm_pairs = outputs / "pairs-sfm.txt"
    # sfm_dir = outputs / "sfm_superpoint+superglue"
    # features = outputs / "features.h5"
    # matches = outputs / "matches.h5"


    # Setup the paths
    # dataset = args.dataset
    # images = dataset / "images_upright/"
    images = Path("/Users/shilem2/OneDrive - Medtronic PLC/projects/rgbd_reconstruction/data/work_volume_data/20240327_091906_sample/")

    #
    sift_sfm = Path('/Users/shilem2/OneDrive - Medtronic PLC/projects/rgbd_reconstruction/Hierarchical-Localization/outputs/work_volume/based_on_demo/sfm_superpoint+superglue/')

    # outputs = args.outputs  # where everything will be saved
    outputs = Path("../outputs/work_volume/based_on_loftr/")
    shutil.rmtree(outputs.as_posix(), ignore_errors=True)
    outputs.mkdir(parents=True)

    # parameters
    num_covis = 20
    num_loc = 50

    reference_sfm = outputs / "sfm_loftr"  # the SfM model we will build
    sfm_pairs = (outputs / f"pairs-db-covis{num_covis}.txt")  # top-k most covisible in SIFT model
    loc_pairs = (outputs / f"pairs-query-netvlad{num_loc}.txt")  # top-k retrieved by NetVLAD
    results = outputs / f"Aachen-v1.1_hloc_loftr_netvlad{num_loc}.txt"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    # list the standard configurations available
    logger.info("Configs for dense feature matchers:\n%s", pformat(match_dense.confs))

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    matcher_conf = match_dense.confs["loftr_aachen"]

    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=num_covis)
    features, sfm_matches = match_dense.main(matcher_conf, sfm_pairs, images, outputs, max_kps=8192, overwrite=False)

    triangulation.main(reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches)

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )
    features, loc_matches = match_dense.main(
        matcher_conf,
        loc_pairs,
        images,
        outputs,
        features=features,
        max_kps=None,
        matches=sfm_matches,
    )


    # run incremental Structure-From-Motion
    sfm_dir = outputs / "sfm_superpoint+superglue"
    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True)
    fig.show()

    visualization.visualize_sfm_2d(model, images, color_by="visibility", n=2)




    # localize_sfm.main(
    #     reference_sfm,
    #     dataset / "queries/*_time_queries_with_intrinsics.txt",
    #     loc_pairs,
    #     features,
    #     loc_matches,
    #     results,
    #     covisibility_clustering=False,
    # )  # not required with loftr




    pass


if __name__ == '__main__':

    # rgb_reconstruction_based_on_demo()
    # rgb_reconstruction_colmap_dense()
    rgb_reconstruction_semi_dense_pipeline_loftr()

    pass