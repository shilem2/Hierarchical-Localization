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

def main():

    # Setup

    os.chdir('..')

    images = Path("datasets/sacre_coeur")
    outputs = Path("outputs/demo/")
    # !rm - rf $outputs
    shutil.rmtree(outputs.as_posix(), ignore_errors=True)
    sfm_pairs = outputs / "pairs-sfm.txt"
    loc_pairs = outputs / "pairs-loc.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]

    # 3D mapping
    references = [p.relative_to(images).as_posix() for p in (images / "mapping/").iterdir()]
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

    # Localization
    url = "https://upload.wikimedia.org/wikipedia/commons/5/53/Paris_-_Basilique_du_Sacr%C3%A9_Coeur%2C_Montmartre_-_panoramio.jpg"
    # try other queries by uncommenting their url
    # url = "https://upload.wikimedia.org/wikipedia/commons/5/59/Basilique_du_Sacr%C3%A9-C%C5%93ur_%285430392880%29.jpg"
    # url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/Sacr%C3%A9_C%C5%93ur_at_night%21_%285865355326%29.jpg"
    query = "query/night.jpg"
    # !mkdir - p $images / query & & wget $url - O $images /$query - q
    plot_images([read_image(images / query)], dpi=75)

    extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)

    import pycolmap
    from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

    camera = pycolmap.infer_camera_from_image(images / query)
    ref_ids = [model.find_image_with_name(r).image_id for r in references]
    conf = {
        "estimation": {"ransac": {"max_error": 12}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    }
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

    print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
    visualization.visualize_loc_from_log(images, query, log, model)

    pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
    viz_3d.plot_camera_colmap(
        fig, pose, camera, color="rgba(0,255,0,0.5)", name=query, fill=True
    )
    # visualize 2D-3D correspodences
    inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log["points3D_ids"])[ret["inliers"]]])
    viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
    fig.show()

    pass


if __name__ == '__main__':

    main()

    pass