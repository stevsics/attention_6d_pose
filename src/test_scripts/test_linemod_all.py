import tensorflow as tf
import numpy as np
import glob

import os
from tensorflow_graphics.geometry.transformation import quaternion

# import open3d as o3d
from plyfile import PlyData

FLAGS = tf.app.flags.FLAGS

# training FLAGS (used only to instantiate Estimator class, values not important)

tf.app.flags.DEFINE_integer("steps", 100000, "Training steps")

tf.app.flags.DEFINE_integer(
    "sync_replicas", -1,
    "If > 0, use SyncReplicasOptimizer and use this many replicas per sync.")

linemod_cls_names=['ape','cam','cat','duck','glue','iron','phone',
                    'benchvise','can','driller','eggbox','holepuncher','lamp']

linemod_cls_names_occlusion = ['ape','cat','duck','glue','can','driller','eggbox','holepuncher']

crop_dimensions = {
    "x_crop_dim": np.array([-76, 76]),
    "y_crop_dim": np.array([-76, 76])
}

image_params = {
    "img_size": [640, 480],
}
R_init = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

scaling_2_mm = 1000.0


def read_ply_file(file_name):

    with open(file_name, 'rb') as f:
        mesh = PlyData.read(f)
        vertices_model_x = np.expand_dims(mesh['vertex']['x'], axis=-1)
        vertices_model_y = np.expand_dims(mesh['vertex']['y'], axis=-1)
        vertices_model_z = np.expand_dims(mesh['vertex']['z'], axis=-1)

    return np.concatenate((vertices_model_x, vertices_model_y, vertices_model_z), axis=-1) * scaling_2_mm


def read_meshes(mesh_folder):

    mesh_vertices_list = []

    for it_mesh_file in linemod_cls_names:

        it_mesh_file_full = os.path.join(mesh_folder, it_mesh_file + '_color_4000.ply')
        vertices_model = read_ply_file(it_mesh_file_full)

        vertices_model = tf.convert_to_tensor(vertices_model, dtype=tf.float32)
        init_rot_quat = tf.cast(quaternion.from_rotation_matrix(R_init), tf.float32)
        vertices_model = quaternion.rotate(vertices_model, init_rot_quat)

        vertices_model = tf.expand_dims(vertices_model, axis=0)

        mesh_vertices_list.append(vertices_model)

    return  tf.concat(mesh_vertices_list, axis=0)


def read_keypoint_model(mesh_folder):

    keypoints_list = np.zeros((len(linemod_cls_names), 500, 3))

    for it_mesh_file in range(len(linemod_cls_names)):
        it_mesh_file_full = os.path.join(mesh_folder, linemod_cls_names[it_mesh_file] + '_500.ply')

        keypoints_3D_model = read_ply_file(it_mesh_file_full)

        keypoints_list[it_mesh_file, :, :] = keypoints_3D_model

    return keypoints_list


def create_input_fn_tfrecord(dataset, folder_tfrecord, mesh_folder, batch_size):

    tfrecord_files = []

    it_obj_path = os.path.join(folder_tfrecord, dataset)
    print(it_obj_path)
    file_names = sorted(glob.glob(os.path.join(it_obj_path, '*.tfrecord')))
    for file_it in file_names:
        tfrecord_files.append(os.path.join(it_obj_path, file_it))
    print(tfrecord_files)

    def input_fn():
        """input_fn for tf.estimator.Estimator."""

        meshes_list = read_meshes(mesh_folder)
        lambda_mask_scaling = 1.4

        def parser(tfrecord_file):

            features = {
                "img": tf.FixedLenFeature([], tf.string),
                "cls_indexes": tf.VarLenFeature(dtype=tf.int64),
                "obj_num": tf.FixedLenFeature([1], tf.int64),
                "pos": tf.VarLenFeature(dtype=tf.float32),
                "quat": tf.VarLenFeature(dtype=tf.float32),
                "init_pose": tf.FixedLenFeature([3], tf.float32),
                "init_quat": tf.FixedLenFeature([4], tf.float32),
                "K_init_all": tf.FixedLenFeature([13 * 3 * 3], tf.float32),
            }

            fs = tf.parse_single_example(tfrecord_file, features=features)

            cls_indexes = tf.sparse.to_dense(fs["cls_indexes"])
            obj_num = fs["obj_num"]
            pos_all = tf.reshape(tf.sparse.to_dense(fs["pos"]), [-1, 3])
            quat_all = tf.reshape(tf.sparse.to_dense(fs["quat"]), [-1, 4])

            rand_obj_ind = tf.random_uniform([1], minval=[0.0], maxval=tf.cast(obj_num, dtype=tf.float32))
            rand_obj_ind = tf.cast(tf.floor(rand_obj_ind), dtype=tf.int32)
            rand_obj_ind = cls_indexes[rand_obj_ind[0]]

            cls_indexes_one_hot = tf.one_hot(cls_indexes, len(linemod_cls_names))
            cls_indexes_one_hot_obj = cls_indexes_one_hot[:, rand_obj_ind]

            cls_indexes_ind = tf.argmax(cls_indexes_one_hot_obj)

            quat = quat_all[cls_indexes_ind, :]
            pos = pos_all[cls_indexes_ind, :] * scaling_2_mm

            image_decoded = tf.image.decode_png(fs["img"], channels=3)

            vertices_model = meshes_list[rand_obj_ind]

            # init pose
            pos_init = fs["init_pose"] * scaling_2_mm

            quat_init = fs["init_quat"]

            K_init_all = tf.reshape(fs["K_init_all"], [-1, 3, 3])
            camera_intrinsics_tensor = K_init_all[cls_indexes_ind, :]

            # computing image center, by computing projection of the center
            im_center = tf.matmul(tf.expand_dims(pos_init, axis=0), tf.transpose(camera_intrinsics_tensor))
            im_center = tf.squeeze(tf.div(im_center, im_center[:, -1]))

            # computing crop area
            transformed_corners = quaternion.rotate(vertices_model, quat_init) + pos_init

            # computing scaling and crop size based on extreme projection points
            corners_projected_cv = tf.matmul(transformed_corners, tf.transpose(camera_intrinsics_tensor))
            corners_projected_cv = tf.div(corners_projected_cv, tf.expand_dims(corners_projected_cv[:, -1], axis=-1))

            y_min = tf.reduce_min(corners_projected_cv[:, 1], axis=0)
            y_max = tf.reduce_max(corners_projected_cv[:, 1], axis=0)
            x_min = tf.reduce_min(corners_projected_cv[:, 0], axis=0)
            x_max = tf.reduce_max(corners_projected_cv[:, 0], axis=0)

            distance_to_center = tf.stack([
                tf.abs(y_min - im_center[1]),
                tf.abs(y_max - im_center[1]),
                tf.abs(x_min - im_center[0]),
                tf.abs(x_max - im_center[0]),
            ], axis=-1)

            crop_width_half = lambda_mask_scaling * tf.reduce_max(distance_to_center, axis=0)

            bb_xy_bounds = tf.convert_to_tensor([im_center[1] - crop_width_half,
                                                 im_center[0] - crop_width_half,
                                                 im_center[1] + crop_width_half,
                                                 im_center[0] + crop_width_half,
                                                ])

            bb_box_scaling = tf.convert_to_tensor([image_params["img_size"][1], image_params["img_size"][0],
                                                   image_params["img_size"][1], image_params["img_size"][0]],
                                                  dtype=tf.float32)

            bb_box_normalized = tf.div(bb_xy_bounds, bb_box_scaling)

            img_cropped = tf.image.crop_and_resize(tf.expand_dims(image_decoded, axis=0),
                                                   tf.expand_dims(bb_box_normalized, axis=0),
                                                   box_ind=[0],
                                                   crop_size=[-crop_dimensions["y_crop_dim"][0] + crop_dimensions["y_crop_dim"][1],
                                                              -crop_dimensions["x_crop_dim"][0] + crop_dimensions["x_crop_dim"][1]])

            img_cropped = tf.squeeze(img_cropped)

            labels = {
                "pos": tf.cast(pos, dtype=tf.float32),
                "quat": tf.cast(quat, dtype=tf.float32)
            }

            fs = {
                "img": img_cropped,
                "pos": tf.cast(pos, dtype=tf.float32),
                "quat": tf.cast(quat, dtype=tf.float32),
                "pos_init": tf.cast(pos_init, dtype=tf.float32),
                "quat_init": tf.cast(quat_init, dtype=tf.float32),
                "object_ind": [rand_obj_ind],
                "cam_mat": camera_intrinsics_tensor,
            }

            return fs, labels

        dataset = tf.data.TFRecordDataset(tfrecord_files)

        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=32)

        return dataset

    return input_fn


def default_hparams():
    hparams = tf.contrib.training.HParams(
        batch_size=32,
        weight_decay=5e-4,
        learning_rate=1.0e-2,
        keypoints_list=[0.0],
    )
    return hparams


def eval_checkpoint(predict_func, eval_options):
    hparams = default_hparams()

    keypoints_list = read_keypoint_model(eval_options["mesh_folder"])
    hparams.keypoints_list = keypoints_list
    hparams.mesh_folder = eval_options["mesh_folder"]

    for it_obj in range(len(linemod_cls_names)):

        if (eval_options["eval_occlusion"]) and (not linemod_cls_names[it_obj] in linemod_cls_names_occlusion):
            continue

        eval_options["obj_name"] = linemod_cls_names[it_obj]
        predict_func(steps=FLAGS.steps, hparams=hparams, sync_replicas=FLAGS.sync_replicas,
                     input_fn=create_input_fn_tfrecord, external_options=eval_options)

