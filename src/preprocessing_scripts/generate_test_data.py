import tensorflow as tf
import numpy as np
import os
import cv2
from pyrr import Quaternion
import copy

object_names = ['ape','cam','cat','duck','glue','iron','phone',
                'benchvise','can','driller','eggbox','holepuncher','lamp']

object_names_occlusion = ['ape','cat','duck','glue','can','driller','eggbox','holepuncher']

object_indeces = [it for it in range(len(object_names))]

camera_intrinsic_matrix_syn = np.array([[700., 0., 320.],
                                        [0., 700., 240.],
                                        [0., 0., 1.]])

camera_intrinsic_matrix_real = np.array([[572.41140, 0.       , 325.26110],
                                     [0.      , 573.57043, 242.04899],
                                     [0.      , 0.       , 1.       ]])

R_init = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

def save_tf_record_file(out_folder, it_tf_record, examples):
    tf_num = "%06d" % it_tf_record
    tfrecord_file_out = os.path.join(out_folder, tf_num + '.tfrecord')
    with tf.python_io.TFRecordWriter(tfrecord_file_out) as writer:
        for it_save in range(len(examples)):
            it_example = examples[it_save]
            writer.write(it_example.SerializeToString())

    return None


def proccess_real_data_occlusion_linemod(in_folder_name, linemod_folder, init_pose_folder_name, out_folder_name, each_object_separate=False):

    out_folder_name = os.path.join(out_folder_name, 'linemod_occlusion')

    if not os.path.exists(out_folder_name):
        os.mkdir(out_folder_name)

    for it_obj in range(len(object_names)):

        if not object_names[it_obj] in object_names_occlusion:
            continue

        print("Object name: " + object_names[it_obj] + ", object index num: " + str(it_obj))

        folder_poses = os.path.join(in_folder_name, 'blender_poses', object_names[it_obj])
        folder_images = os.path.join(in_folder_name, 'RGB-D', 'rgb_noseg')
        out_folder_name_obj = os.path.join(out_folder_name, object_names[it_obj])

        if not os.path.exists(out_folder_name_obj):
            os.mkdir(out_folder_name_obj)

        occlusion_test_file = os.path.join(linemod_folder, object_names[it_obj], 'test_occlusion.txt')
        inds = np.loadtxt(occlusion_test_file, np.str)
        inds = [int(os.path.basename(ind).replace('.jpg', '')) for ind in inds]

        it_tf_record = 0
        examples = []

        for it_img, it_indx in enumerate(inds):
            if each_object_separate:
                if not (len(examples) == 0):
                    print(it_img)
                    save_tf_record_file(out_folder_name_obj, it_tf_record, examples)
                    it_tf_record += 1
                    examples = []
            else:
                if (it_img % 100) == 0:
                    if not (len(examples) == 0):
                        print(it_img)
                        save_tf_record_file(out_folder_name_obj, it_tf_record, examples)
                        it_tf_record += 1
                    examples = []

            it_obj_name_pose = "pose" + str(it_indx) + ".npy"
            it_obj_name_img = "color_" + "%05d" % it_indx + ".png"

            poses_file = os.path.join(folder_poses, it_obj_name_pose)
            image_file = os.path.join(folder_images, it_obj_name_img)

            pos = np.zeros((1, 3))
            quat = np.zeros((1, 4))

            if os.path.exists(poses_file):
                data = np.load(poses_file)
            else:
                data = np.array([[1.0, 0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0, 1.0],
                                 [0.0, 0.0, 1.0, 1.0], ])

            R_mat = data[:3, :3]
            pos[0, :] = data[:3, 3]
            quat[0, :] = Quaternion.from_matrix(R_mat)

            cls_indexes = [object_indeces[it_obj]]
            cls_indexes_num = [1]

            # read image
            img_name = os.path.join(folder_images, image_file)
            out_img = cv2.imread(img_name)

            encode_image = tf.compat.as_bytes(cv2.imencode(".png", out_img)[1].tostring())

            # read init pose
            it_obj_name_init = "%06d" % it_img + "_predict.npy"
            init_pose_file = os.path.join(init_pose_folder_name, object_names[it_obj], it_obj_name_init)

            predict_data = np.load(init_pose_file, allow_pickle='TRUE').item()
            Rt_mat_init = predict_data["pose_pred"]
            R_mat_init = np.matmul(Rt_mat_init[:3, :3], R_init)
            quat_init = Quaternion.from_matrix(R_mat_init)
            pos_init = Rt_mat_init[:, 3]

            if np.isnan(pos_init[0]):
                pos_init = np.array([0.0, 0.0, 10.0])
                quat_init = np.array([0.0, 0.0, 0.0, 1.0])

            if np.linalg.norm(pos_init) > 10.0:
                pos_init = np.array([0.0, 0.0, 10.0])
                quat_init = np.array([0.0, 0.0, 0.0, 1.0])

            num_of_objects = 13
            K_init_all = np.zeros((num_of_objects, 3, 3))

            for it_img_obj in range(num_of_objects):
                K_init_all[it_img_obj, :, :] = copy.copy(camera_intrinsic_matrix_real)

            feature = {
                "init_pose": tf.train.Feature(float_list=tf.train.FloatList(value=pos_init)),
                "init_quat": tf.train.Feature(float_list=tf.train.FloatList(value=quat_init)),
                "cls_indexes": tf.train.Feature(int64_list=tf.train.Int64List(value=cls_indexes)),
                "obj_num": tf.train.Feature(int64_list=tf.train.Int64List(value=cls_indexes_num)),
                "pos": tf.train.Feature(float_list=tf.train.FloatList(value=pos.reshape(-1))),
                "quat": tf.train.Feature(float_list=tf.train.FloatList(value=quat.reshape(-1))),
                "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encode_image])),
                "K_init_all": tf.train.Feature(float_list=tf.train.FloatList(value=K_init_all.reshape(-1))),
            }
            tf_record_example = tf.train.Example(features=tf.train.Features(feature=feature))
            examples.append(tf_record_example)

        if not (len(examples) == 0):
            save_tf_record_file(out_folder_name_obj, it_tf_record, examples)


def proccess_real_data_linemod(linemod_folder, init_pose_folder_name, out_folder_name, each_object_separate=False):

    out_folder_name = os.path.join(out_folder_name, 'linemod')

    if not os.path.exists(out_folder_name):
        os.mkdir(out_folder_name)

    for it_obj in range(len(object_names)):

        print("Object name: " + object_names[it_obj] + ", object index num: " + str(it_obj))

        in_folder_name_obj = os.path.join(linemod_folder, object_names[it_obj])
        folder_images = os.path.join(in_folder_name_obj, "JPEGImages")
        in_folder_name_mask_obj = os.path.join(in_folder_name_obj, "mask")
        folder_poses = os.path.join(in_folder_name_obj, "pose")
        out_folder_name_obj = os.path.join(out_folder_name, object_names[it_obj])

        if not os.path.exists(out_folder_name_obj):
            os.mkdir(out_folder_name_obj)

        occlusion_test_file = os.path.join(linemod_folder, object_names[it_obj], 'test.txt')
        inds = np.loadtxt(occlusion_test_file, np.str)
        inds = [int(os.path.basename(ind).replace('.jpg', '')) for ind in inds]

        it_tf_record = 0
        examples = []

        for it_img, it_indx in enumerate(inds):
            if each_object_separate:
                if not (len(examples) == 0):
                    print(it_img)
                    save_tf_record_file(out_folder_name_obj, it_tf_record, examples)
                    it_tf_record += 1
                    examples = []
            else:
                if (it_img % 100) == 0:
                    if not (len(examples) == 0):
                        print(it_img)
                        save_tf_record_file(out_folder_name_obj, it_tf_record, examples)
                        it_tf_record += 1
                    examples = []

            it_obj_name_pose = "pose" + str(it_indx) + ".npy"
            it_obj_name_img = "%06d" % it_indx + ".jpg"

            poses_file = os.path.join(folder_poses, it_obj_name_pose)
            image_file = os.path.join(folder_images, it_obj_name_img)

            pos = np.zeros((1, 3))
            quat = np.zeros((1, 4))

            if os.path.exists(poses_file):
                data = np.load(poses_file)
            else:
                data = np.array([[1.0, 0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0, 1.0],
                                 [0.0, 0.0, 1.0, 1.0], ])

            R_mat = data[:3, :3]
            pos[0, :] = data[:3, 3]
            quat[0, :] = Quaternion.from_matrix(R_mat)

            cls_indexes = [object_indeces[it_obj]]
            cls_indexes_num = [1]

            # read image
            img_name = os.path.join(folder_images, image_file)
            out_img = cv2.imread(img_name)

            encode_image = tf.compat.as_bytes(cv2.imencode(".png", out_img)[1].tostring())

            # read init pose
            it_obj_name_init = "%06d" % it_img + "_predict.npy"
            init_pose_file = os.path.join(init_pose_folder_name, object_names[it_obj], it_obj_name_init)

            predict_data = np.load(init_pose_file, allow_pickle='TRUE').item()
            Rt_mat_init = predict_data["pose_pred"]
            R_mat_init = np.matmul(Rt_mat_init[:3, :3], R_init)
            quat_init = Quaternion.from_matrix(R_mat_init)
            pos_init = Rt_mat_init[:, 3]

            if np.isnan(pos_init[0]):
                pos_init = np.array([0.0, 0.0, 10.0])
                quat_init = np.array([0.0, 0.0, 0.0, 1.0])

            if np.linalg.norm(pos_init) > 10.0:
                pos_init = np.array([0.0, 0.0, 10.0])
                quat_init = np.array([0.0, 0.0, 0.0, 1.0])

            num_of_objects = 13
            K_init_all = np.zeros((num_of_objects, 3, 3))

            for it_img_obj in range(num_of_objects):
                K_init_all[it_img_obj, :, :] = copy.copy(camera_intrinsic_matrix_real)

            feature = {
                "init_pose": tf.train.Feature(float_list=tf.train.FloatList(value=pos_init)),
                "init_quat": tf.train.Feature(float_list=tf.train.FloatList(value=quat_init)),
                "cls_indexes": tf.train.Feature(int64_list=tf.train.Int64List(value=cls_indexes)),
                "obj_num": tf.train.Feature(int64_list=tf.train.Int64List(value=cls_indexes_num)),
                "pos": tf.train.Feature(float_list=tf.train.FloatList(value=pos.reshape(-1))),
                "quat": tf.train.Feature(float_list=tf.train.FloatList(value=quat.reshape(-1))),
                "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encode_image])),
                "K_init_all": tf.train.Feature(float_list=tf.train.FloatList(value=K_init_all.reshape(-1))),
            }
            tf_record_example = tf.train.Example(features=tf.train.Features(feature=feature))
            examples.append(tf_record_example)

        if not (len(examples) == 0):
            save_tf_record_file(out_folder_name_obj, it_tf_record, examples)


if __name__ == '__main__':

    linemod_occlusion_folder = "../resources/pvnet_data/OCCLUSION_LINEMOD"
    linemod_folder = "../resources/pvnet_data/LINEMOD"
    out_folder = "/media/data_ssd/Datasets/LINEMOD_Processed"
    init_pose_folder_name = "../resources/pvnet_data/init_poses/occlusion"

    proccess_real_data_occlusion_linemod(linemod_occlusion_folder, linemod_folder, init_pose_folder_name, out_folder,
                                         each_object_separate=True)

    init_pose_folder_name = "../resources/pvnet_data/init_poses/linemod"

    proccess_real_data_linemod(linemod_folder, init_pose_folder_name, out_folder, each_object_separate=True)

