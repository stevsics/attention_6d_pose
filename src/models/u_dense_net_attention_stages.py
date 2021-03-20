import tensorflow as tf
import numpy as np
from pyrr import Quaternion
import os

from tensorflow_graphics.geometry.transformation import quaternion

from models.dense_net import relu, conv_costum_init_layer
from models.diff_renderer import MyDiffRendererStages
from models.dense_net import UDenseNet


FLAGS = tf.app.flags.FLAGS

crop_dimensions = {
    "x_crop_dim": np.array([-76, 76]),
    "y_crop_dim": np.array([-76, 76])
}

image_params = {
    "img_size": [640, 480],
}

dense_net_params = {
    "grow_rate": 16,
}

model_params = {
    "grid_size": 38,
}

model_names = ['ape','cam','cat','duck','glue','iron','phone',
               'benchvise','can','driller','eggbox','holepuncher','lamp']

scaling_2_mm = 1000.0
far_near_cut = [10000.0, 100.0]

R_init = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


def predict_the_pose(pos_init, quat_init, xy_pred, z_pred, quat_pred, img_scaling, cam_mat):

    img_center = tf.matmul(tf.expand_dims(pos_init, axis=1), cam_mat, transpose_b=True)
    img_center = tf.reduce_sum(img_center, axis=1)
    img_center = tf.div(img_center, tf.expand_dims(img_center[:, -1], axis=1))

    xy_image = img_center[:, :2] + xy_pred / tf.expand_dims(img_scaling, axis=-1)

    z_model = pos_init[:, 2]

    x_model_cam_frame = (xy_image[:, 0] - cam_mat[:, 0, 2]) / cam_mat[:, 0, 0]
    x_model_cam_frame = tf.expand_dims(tf.multiply(x_model_cam_frame, z_model), axis=-1)
    y_model_cam_frame = (xy_image[:, 1] - cam_mat[:, 1, 2]) / cam_mat[:, 1, 1]
    y_model_cam_frame = tf.expand_dims(tf.multiply(y_model_cam_frame, z_model), axis=-1)

    z_model = tf.expand_dims(z_model, axis=-1)
    pose_model_lateral = tf.concat((x_model_cam_frame, y_model_cam_frame, z_model), axis=-1)

    z_out = z_pred / 3000.0
    pos_out = tf.multiply(tf.exp(z_out), pose_model_lateral)

    quat = quat_pred / 10.0
    quat_len = tf.expand_dims(tf.norm(quat, axis=-1), axis=-1)
    quat_out = tf.div(quat, quat_len)
    quat_out = quaternion.multiply(quat_out, quat_init)

    return pos_out, quat_out


class ModelUDenseNetAttentionStages(tf.keras.layers.Layer):

    def __init__(self,  mode, weight_decay, dense_net_params, size_params, mesh_folder):
        super(ModelUDenseNetAttentionStages, self).__init__()

        if mode == tf.estimator.ModeKeys.TRAIN:
            training = True
        else:
            training = False
        self.training = training

        self.stages = []
        for it_stage in range(4):
            stage = ModelUDenseNetAttentionLayer(weight_decay, dense_net_params, size_params, mesh_folder, self.training)
            self.stages.append(stage)

    def call(self, input):

        img_input = input[0]
        quat_pos_ind_cam_mat = input[1]

        pose_prediction = []
        quat_prediction = []

        quat_init, pos_init, obj_ind, cam_mat = tf.split(quat_pos_ind_cam_mat, (4, 3, 1, 9), axis=-1)
        cam_mat_3x3 = tf.reshape(cam_mat, [-1, 3, 3])
        quat_pos_ind_cam_mat_pos_init = tf.concat((quat_init, pos_init, obj_ind, cam_mat, pos_init, quat_init), axis=-1)

        for it_stage in self.stages:

            quat, pos, obj_ind, cam_mat, pos_init, quat_init = tf.split(quat_pos_ind_cam_mat_pos_init,
                                                                        (4, 3, 1, 9, 3, 4), axis=-1)

            xy_out, z_out, quat_out, im_center, image_scaling = it_stage([img_input, quat_pos_ind_cam_mat_pos_init])
            pos_out, quat_out = predict_the_pose(pos, quat, xy_out, z_out, quat_out, image_scaling, cam_mat_3x3)

            pose_prediction.append(pos_out)
            quat_prediction.append(quat_out)

            quat_pos_ind_cam_mat_pos_init = tf.concat((quat_out, pos_out, obj_ind, cam_mat, pos_init, quat_init), axis=-1)

        return pose_prediction, quat_prediction

class ModelUDenseNetAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, weight_decay, dense_net_params, size_params, mesh_folder, training):
        super(ModelUDenseNetAttentionLayer, self).__init__()

        self.batch_size = size_params["batch_size"]
        self.grid_size = size_params["grid_size"]
        weight_decay = (weight_decay, 0)
        dense_net_params["weight_decay"] = weight_decay

        self.training = training

        models_path = []
        for it_model in model_names:
            models_path.append(os.path.join(mesh_folder, it_model + '_color_4000.ply'))

        self.diff_rend = MyDiffRendererStages(far_near_cut, models_path, R_init, scaling_2_mm, use_texture=True)
        self.concat_join_in = tf.keras.layers.Concatenate(axis=-1)

        self.u_dense_net = UDenseNet('dense_net', dense_net_params, training)
        self.batch_norm_layer = tf.keras.layers.BatchNormalization(name='dense_net_out_bn')

        # xy stream
        self.xy_conv_layer = conv_costum_init_layer(256, 1, 'quat_pos_conv_layer', weight_decay,
                                                    1.0 / self.u_dense_net.features_count)
        self.xy_att = tf.keras.layers.Dense(256, name="quat_pos_att")
        self.xy_conv_out = tf.keras.layers.Dense(2, name="quat_pos_out")
        self.xy_out_batch_norm = tf.keras.layers.BatchNormalization(name='quat_pose_out_bn')

        # quat stream
        self.quat_conv_layer = conv_costum_init_layer(256, 1, 'quat_conv_layer', weight_decay,
                                                      1.0 / self.u_dense_net.features_count)
        self.quat_att = tf.keras.layers.Dense(256, name="quat_att")
        self.quat_out = tf.keras.layers.Dense(4, name="quat_out")
        self.quat_out_batch_norm = tf.keras.layers.BatchNormalization(name='quat_out_bn')

        # z stream
        self.z_conv_layer = conv_costum_init_layer(256, 1, 'z_conv_layer', weight_decay,
                                                   1.0 / self.u_dense_net.features_count)
        self.z_out_att = tf.keras.layers.Dense(256, name="z_out_att")
        self.z_out = tf.keras.layers.Dense(1, name="z_out")
        self.z_out_batch_norm = tf.keras.layers.BatchNormalization(name='z_out_bn')

        self.attention_out_batch_norm_layer = tf.keras.layers.BatchNormalization(name='attention_out_bn')

        # att xy
        self.attention_conv_layer_xy = conv_costum_init_layer(256, 1, 'attention_conv_layer', weight_decay,
                                                              1.0 / self.u_dense_net.features_count)
        self.attention_out_xy = conv_costum_init_layer(1, 1, 'attention_out', weight_decay,
                                                       1.0 / 256)
        self.attention_out_xy_batch_norm = tf.keras.layers.BatchNormalization(name='attention_out_bn')

        # att z
        self.attention_conv_layer_z = conv_costum_init_layer(256, 1, 'attention_conv_layer_z', weight_decay,
                                                             1.0 / self.u_dense_net.features_count)
        self.attention_out_z = conv_costum_init_layer(1, 1, 'attention_out_z', weight_decay,
                                                      1.0 / 256)
        self.attention_out_z_batch_norm = tf.keras.layers.BatchNormalization(name='attention_z_out_bn')

        # att quat
        self.attention_conv_layer_q = conv_costum_init_layer(256, 1, 'attention_conv_layer_quat', weight_decay,
                                                             1.0 / self.u_dense_net.features_count)
        self.attention_out_q = conv_costum_init_layer(1, 1, 'attention_out_quat', weight_decay,
                                                      1.0 / 256)
        self.attention_out_q_batch_norm = tf.keras.layers.BatchNormalization(name='attention_q_out_bn')

    def call(self, input):
        img_input = input[0]
        quat_pos_input = input[1]

        # diff renderer
        diff_renderer_out, cube_vertices_clip, im_center, image_scaling = self.diff_rend(quat_pos_input)
        full_image = tf.zeros((self.batch_size, 2, 2))
        bb_xy_bounds = tf.zeros((self.batch_size, 2, 2))
        diff_renderer_out_zero_mean = tf.subtract(diff_renderer_out, 0.5)

        # dense net 38x38 grid predictions
        joint_feature = self.concat_join_in([img_input, diff_renderer_out_zero_mean])
        ae_out, bottleneck_out, decoder_2_out, decoder_1_out = self.u_dense_net(joint_feature)

        x = self.batch_norm_layer(ae_out, training=self.training)
        x = relu(x)

        # attention out
        attention_bn = self.attention_out_batch_norm_layer(ae_out, training=self.training)
        attention = relu(attention_bn)
        attention = self.attention_conv_layer_xy(attention)
        attention = self.attention_out_xy_batch_norm(attention, training=self.training)
        attention = relu(attention)
        attention_xy = self.attention_out_xy(attention)

        attention = relu(attention_bn)
        attention = self.attention_conv_layer_z(attention)
        attention = self.attention_out_z_batch_norm(attention, training=self.training)
        attention = relu(attention)
        attention_z = self.attention_out_z(attention)

        attention = relu(attention_bn)
        attention = self.attention_conv_layer_q(attention)
        attention = self.attention_out_q_batch_norm(attention, training=self.training)
        attention = relu(attention)
        attention_quat = self.attention_out_q(attention)

        # z out
        attention_map_z = tf.reduce_sum(attention_z, axis=-1)
        attention_map_z = tf.reshape(attention_map_z, shape=[self.batch_size, self.grid_size * self.grid_size])
        attention_map_z = tf.nn.softmax(attention_map_z, axis=-1)
        attention_map_z = tf.reshape(attention_map_z, shape=[self.batch_size, self.grid_size, self.grid_size])
        attention_map_z = tf.expand_dims(attention_map_z, axis=-1)

        z_out = self.z_conv_layer(x)
        z_out = self.z_out_batch_norm(z_out, training=self.training)
        z_out = relu(z_out)
        z_out_weighted = tf.multiply(z_out, attention_map_z)
        z_out_sum = tf.reduce_sum(z_out_weighted, axis=[1, 2])
        z_out = self.z_out_att(z_out_sum)
        z_out = self.z_out(z_out)

        # xy out
        attention_map_xy = tf.reduce_sum(attention_xy, axis=-1)
        attention_map_xy = tf.reshape(attention_map_xy, shape=[self.batch_size, self.grid_size * self.grid_size])
        attention_map_xy = tf.nn.softmax(attention_map_xy, axis=-1)
        attention_map_xy = tf.reshape(attention_map_xy, shape=[self.batch_size, self.grid_size, self.grid_size])
        attention_map_xy = tf.expand_dims(attention_map_xy, axis=-1)

        xy_out = self.xy_conv_layer(x)
        xy_out = self.xy_out_batch_norm(xy_out, training=self.training)
        xy_out = relu(xy_out)
        xy_out_weighted = tf.multiply(xy_out, attention_map_xy)
        xy_out_sum = tf.reduce_sum(xy_out_weighted, axis=[1, 2])
        xy_out = self.xy_att(xy_out_sum)
        xy_out = self.xy_conv_out(xy_out)

        #  quat out
        attention_map_quat = tf.reduce_sum(attention_quat, axis=-1)
        attention_map_quat = tf.reshape(attention_map_quat, shape=[self.batch_size, self.grid_size * self.grid_size])
        attention_map_quat = tf.nn.softmax(attention_map_quat, axis=-1)
        attention_map_quat = tf.reshape(attention_map_quat, shape=[self.batch_size, self.grid_size, self.grid_size])
        attention_map_quat = tf.expand_dims(attention_map_quat, axis=-1)

        quat_out = self.quat_conv_layer(x)
        quat_out = self.quat_out_batch_norm(quat_out, training=self.training)
        quat_out = relu(quat_out)
        qaut_out_weighted = tf.multiply(quat_out, attention_map_quat)
        quat_out_sum = tf.reduce_sum(qaut_out_weighted, axis=[1, 2])
        quat_out = self.quat_att(quat_out_sum)
        quat_out = self.quat_out(quat_out)

        return xy_out, z_out, quat_out, im_center, image_scaling,


def loss_stages(labels, net_out, keypoints_3D_model):

    loss_list = []
    loss_sum = tf.constant(0.0)

    for it_pred_pose, it_pred_quat in zip(net_out['pos'], net_out['quat']) :

        loss_it = 0.5 * stage_loss_keypoints(it_pred_pose, it_pred_quat, labels['pos'], labels['quat'], keypoints_3D_model)
        loss_it = tf.reduce_mean(loss_it)
        loss_list.append(loss_it)
        loss_sum += loss_it

    loss = loss_sum/len(loss_list)

    return loss, loss_list


def stage_loss_keypoints(pos_out, quat_out, pos_gt, quat_gt, keypoints_3D_model):

    keypoints_3D = quaternion.rotate(keypoints_3D_model, tf.expand_dims(quat_gt, axis=-2))
    keypoints_3D_predicted = quaternion.rotate(keypoints_3D_model, tf.expand_dims(quat_out, axis=1))

    keypoints_3D = keypoints_3D + tf.expand_dims(pos_gt, axis=1)
    keypoints_3D_predicted = keypoints_3D_predicted + tf.expand_dims(pos_out, axis=1)

    keypoint_distance = keypoints_3D_predicted - keypoints_3D
    keypoint_distance = tf.norm(keypoint_distance, axis=-1)
    loss_keypoint = tf.reduce_mean(0.5 * tf.square(keypoint_distance), axis=-1)

    return loss_keypoint


def model_fn(features, labels, mode, hparams):

    weight_decay = hparams.weight_decay
    batch_size = tf.shape(features["pos_init"])[0]

    # init pos and quaternion
    pos_init = features["pos_init"]
    quat_init = features["quat_init"]
    obj_ind = features["object_ind"]
    keypoints_3D_model = tf.convert_to_tensor(hparams.keypoints_list)

    keypoints_3D_model_batch = tf.tile(tf.expand_dims(keypoints_3D_model, axis=0), [batch_size, 1, 1, 1])
    keypoints_3D_model = tf.gather_nd(keypoints_3D_model_batch, obj_ind, batch_dims=1)
    keypoints_3D_model = tf.cast(keypoints_3D_model, dtype=tf.float32)

    grid_size = model_params["grid_size"]
    cam_mat = features["cam_mat"]

    quat_pos_indx_cam_mat_input = tf.concat([quat_init, pos_init, tf.cast(obj_ind, dtype=tf.float32),
                                             tf.reshape(cam_mat, [-1, 3*3])], axis=-1)

    size_params = {
        "batch_size": batch_size,
        "grid_size": grid_size,
    }

    model = ModelUDenseNetAttentionStages(mode, weight_decay, dense_net_params, size_params, hparams.mesh_folder)

    img_input = features['img']
    img_input_scaled = tf.subtract(tf.div(tf.cast(img_input, dtype=tf.float32), 256.0), 0.5) # this is ok, cropping does not scale values

    ind = tf.cast(features["object_ind"], dtype=tf.float32)
    ind = tf.expand_dims(ind, axis=1)
    ind = tf.expand_dims(ind, axis=1)
    x_img_dim = crop_dimensions["x_crop_dim"][1] - crop_dimensions["x_crop_dim"][0]
    y_img_dim = crop_dimensions["y_crop_dim"][1] - crop_dimensions["y_crop_dim"][0]
    cls_img_in = tf.ones((batch_size, y_img_dim, x_img_dim, 1)) * ind / 13.0 - 0.5

    output_pos, output_quat = model([img_input_scaled, quat_pos_indx_cam_mat_input, cls_img_in, cam_mat])

    if mode == tf.estimator.ModeKeys.PREDICT:
        labels = {
            "pos": features["pos"],
            "quat": features["quat"],
        }

    net_out = {
        "pos": output_pos,
        "quat": output_quat,
    }

    loss, loss_list = loss_stages(labels, net_out, keypoints_3D_model)

    # summaries
    tf.summary.scalar('loss_stage_1', loss_list[0])
    tf.summary.scalar('loss_stage_2', loss_list[1])
    tf.summary.scalar('loss_stage_3', loss_list[2])
    tf.summary.scalar('loss_stage_4', loss_list[3])

    return {
        "loss": loss,
        "predictions": {
            "pos_out_0": output_pos[0],
            "pos_out_1": output_pos[1],
            "pos_out_2": output_pos[2],
            "pos_out_3": output_pos[3],
            "quat_out_0": output_quat[0],
            "quat_out_1": output_quat[1],
            "quat_out_2": output_quat[2],
            "quat_out_3": output_quat[3],
            "input": features["img"],
            "pos_init": features["pos_init"],
            "quat_init": features["quat_init"],
            "pos_gt": features["pos"],
            "quat_gt": features["quat"],
            "cam_mat": cam_mat,
        },
        "eval_metric_ops": {
        },
        "model_keras": model,
    }

def estimator_model_fn(features, labels, mode, params):
    ret = model_fn(features, labels, mode, params)
    train_op = None
    training_hooks = []

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=ret["predictions"],
        loss=ret["loss"],
        train_op=train_op,
        eval_metric_ops=ret["eval_metric_ops"],
        training_hooks=training_hooks)


def predict_eval_and_save_external(steps, hparams, sync_replicas, input_fn, external_options=None):

    object_to_test = external_options["obj_name"]
    out_folder = external_options["out_folder"]
    ckpt_path = external_options["ckpt_path"]
    dataset_folder = external_options["dataset_folder"]
    mesh_folder = external_options["mesh_folder"]

    run_config = tf.estimator.RunConfig(
        keep_checkpoint_every_n_hours=0.5,
        save_checkpoints_secs=180,
        save_summary_steps=50)

    estimator = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        params=hparams, config=run_config)

    print("loading model: ", ckpt_path)

    input_fn = input_fn(dataset=object_to_test, folder_tfrecord=dataset_folder, mesh_folder=mesh_folder, batch_size=1)
    predictions = estimator.predict(input_fn=input_fn, checkpoint_path=ckpt_path)

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    out_folder_object = os.path.join(out_folder, object_to_test)
    if not os.path.exists(out_folder_object):
        os.mkdir(out_folder_object)

    it_img = 0

    for item in predictions:

        pos_gt = item["pos_gt"]
        quat_gt = item["quat_gt"]

        pos_init = item["pos_init"]
        quat_init = item["quat_init"]

        pred_0 = item["pos_out_0"]
        pred_1 = item["pos_out_1"]
        pred_2 = item["pos_out_2"]
        pred_3 = item["pos_out_3"]

        quat_0 = item["quat_out_0"]
        quat_1 = item["quat_out_1"]
        quat_2 = item["quat_out_2"]
        quat_3 = item["quat_out_3"]

        pose_pred_it = np.zeros((3, 4))
        pose_pred_it[:, 3] = pred_3
        quat_pred = Quaternion(quat_3)
        R_pred = np.array(quat_pred.matrix33)
        pose_pred_it[:3, :3] = R_pred

        pose_pred_it_0 = np.zeros((3, 4))
        quat_pred = Quaternion(quat_0)
        R_pred = np.array(quat_pred.matrix33)
        pose_pred_it_0[:3, :3] = R_pred
        pose_pred_it_0[:, 3] = pred_0

        pose_pred_it_1 = np.zeros((3, 4))
        quat_pred = Quaternion(quat_1)
        R_pred = np.array(quat_pred.matrix33)
        pose_pred_it_1[:3, :3] = R_pred
        pose_pred_it_1[:, 3] = pred_1

        pose_pred_it_2 = np.zeros((3, 4))
        quat_pred = Quaternion(quat_2)
        R_pred = np.array(quat_pred.matrix33)
        pose_pred_it_2[:3, :3] = R_pred
        pose_pred_it_2[:, 3] = pred_2


        pose_gt_it = np.zeros((3, 4))
        pose_gt_it[:, 3] = pos_gt
        quat_gt = Quaternion(quat_gt)
        R_gt = np.array(quat_gt.matrix33)
        pose_gt_it[:3, :3] = R_gt

        if it_img % 100 == 0:
            print(it_img)

        object_to_test = "%06d" % it_img
        out_file = os.path.join(out_folder_object, object_to_test + "_predict.npy")
        out_dict = {
            'pose_pred': pose_pred_it,
            'pose_gt': pose_gt_it,
            'pose_pred_0': pose_pred_it_0,
            'pose_pred_1': pose_pred_it_1,
            'pose_pred_2': pose_pred_it_2,
        }
        np.save(out_file, out_dict)

        it_img += 1
