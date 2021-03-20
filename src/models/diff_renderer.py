import tensorflow as tf
import numpy as np
from plyfile import PlyData

import dirt
import dirt.lighting as lighting

from tensorflow_graphics.geometry.transformation import quaternion


crop_dimensions = {
    "x_crop_dim": np.array([-76, 76]),
    "y_crop_dim": np.array([-76, 76])
}

image_params = {
    "img_size": [152, 152],
}


def read_vertices_faces(ply_file):

    with open(ply_file, 'rb') as f:
        mesh = PlyData.read(f)
        vertices_model_x = np.expand_dims(mesh['vertex']['x'], axis=-1)
        vertices_model_y = np.expand_dims(mesh['vertex']['y'], axis=-1)
        vertices_model_z = np.expand_dims(mesh['vertex']['z'], axis=-1)

        faces = mesh['face']['vertex_indices']
        faces = np.vstack(faces)

    return np.concatenate((vertices_model_x, vertices_model_y, vertices_model_z), axis=-1), faces


def read_vertices_faces_textures(ply_file):

    with open(ply_file, 'rb') as f:
        mesh = PlyData.read(f)
        vertices_model_x = np.expand_dims(mesh['vertex']['x'], axis=-1)
        vertices_model_y = np.expand_dims(mesh['vertex']['y'], axis=-1)
        vertices_model_z = np.expand_dims(mesh['vertex']['z'], axis=-1)

        vertices_color_x = np.expand_dims(mesh['vertex']['red'], axis=-1)
        vertices_color_y = np.expand_dims(mesh['vertex']['green'], axis=-1)
        vertices_color_z = np.expand_dims(mesh['vertex']['blue'], axis=-1)

        faces = mesh['face']['vertex_indices']
        faces = np.vstack(faces)

    return np.concatenate((vertices_model_x, vertices_model_y, vertices_model_z), axis=-1), faces, \
           np.concatenate((vertices_color_x, vertices_color_y, vertices_color_z), axis=-1)


class MyDiffRenderer(tf.keras.layers.Layer):
    def __init__(self, far_near_cut, model_files, init_rot, scaling, use_texture=False):
        super(MyDiffRenderer, self).__init__()
        self.init_rot_quat = tf.cast(quaternion.from_rotation_matrix(init_rot), tf.float32)
        self.lambda_mask_scaling = 1.4

        self.vertices_all = []
        self.faces_all = []
        self.vertex_colors_all = []

        for it_obj in range(len(model_files)):

            if use_texture:
                vertices, faces, vertices_color = read_vertices_faces_textures(model_files[it_obj])
            else:
                vertices, faces = read_vertices_faces(model_files[it_obj])
            vertices *= scaling
            faces = tf.convert_to_tensor(faces, dtype=tf.int32)

            # init rotation of model
            vertices = tf.convert_to_tensor(vertices, dtype=tf.float32)
            vertices = quaternion.rotate(vertices, self.init_rot_quat)

            # Build the scene geometry, which is just an axis-aligned cube centred at the origin in world space
            # We replicate vertices that are shared, so normals are effectively per-face instead of smoothed
            # vertices, faces = build_cube()
            vertices, faces = lighting.split_vertices_by_face(vertices, faces)

            min_vertices = tf.reduce_min(vertices, axis=-2)
            max_vertices = tf.reduce_max(vertices, axis=-2)

            # generate 3D object coordinates by setting vertex color to color corresponding to 3D bounding box
            if use_texture:
                vertex_colors = tf.cast(vertices_color, dtype=tf.float32) / 255.0
            else:
                vertex_colors = vertices - min_vertices  # origin to [0, 0, 0]
                vertex_colors = tf.div(vertex_colors, -min_vertices + max_vertices)  # scale by BB size

            self.vertices_all.append(tf.expand_dims(vertices, axis=0))
            self.faces_all.append(tf.expand_dims(faces, axis=0))
            self.vertex_colors_all.append(tf.expand_dims(vertex_colors, axis=0))

        self.vertices_all = tf.concat(self.vertices_all, axis=0)
        self.faces_all = tf.concat(self.faces_all, axis=0)
        self.vertex_colors_all = tf.concat(self.vertex_colors_all, axis=0)

        # Transform vertices from camera to clip space
        far = far_near_cut[0]
        near =  far_near_cut[1]

        projection_matrix_mul_cam_matrix = [
            [1. / (image_params["img_size"][0] / 2), 0., 0.],
            [0., 1. / (image_params["img_size"][1] / 2), 0.],
            [0., 0., 0.0]
        ]

        projection_matrix_add = [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ]

        self.projection_matrix_mul_cam_matrix = tf.convert_to_tensor(projection_matrix_mul_cam_matrix, dtype=tf.float32)
        self.projection_matrix_add = tf.convert_to_tensor(projection_matrix_add, dtype=tf.float32)

        projection_centering_vec = [1. / (image_params["img_size"][0] / 2), 1. / (image_params["img_size"][1] / 2), 0., 0.]
        self.projection_centering_vec = tf.convert_to_tensor(projection_centering_vec, dtype=tf.float32)

        self.projection_centering_matrix_base = tf.convert_to_tensor([
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ], dtype=tf.float32)


    def call(self, x):

        quat, pos, obj_ind, cam_mat = tf.split(x, (4, 3, 1, 9), axis=-1)
        batch_size = tf.shape(x)[0]
        obj_ind = tf.cast(obj_ind, dtype=tf.int32)
        cam_mat = tf.reshape(cam_mat, [-1, 3, 3])

        # get specific object vertices, faces, vertex colors (textures)
        vertices = tf.gather_nd(self.vertices_all, obj_ind)
        faces = tf.gather_nd(self.faces_all, obj_ind)
        vertex_colors = tf.gather_nd(self.vertex_colors_all, obj_ind)

        # computing image center, by computing projection of the center
        im_center = tf.matmul(tf.expand_dims(pos, axis=1), cam_mat, transpose_b=True)
        im_center = tf.reduce_sum(im_center, axis=1)
        im_center = tf.div(im_center, tf.expand_dims(im_center[:, -1], axis=1))

        # computing transformation of the vertices
        quat_broadcast = tf.expand_dims(quat, axis=-2)
        pos_broadcast = tf.expand_dims(pos, axis=-2)

        transformed_corners = quaternion.rotate(vertices, quat_broadcast) + pos_broadcast

        # computing scaling and crop size based on extreme projection points
        corners_projected_cv = tf.matmul(transformed_corners, cam_mat, transpose_b=True)
        corners_projected_cv = tf.div(corners_projected_cv, tf.expand_dims(corners_projected_cv[:, :, -1], axis=-1))

        y_min = tf.reduce_min(corners_projected_cv[:, :, 1], axis=1)
        y_max = tf.reduce_max(corners_projected_cv[:, :, 1], axis=1)
        x_min = tf.reduce_min(corners_projected_cv[:, :, 0], axis=1)
        x_max = tf.reduce_max(corners_projected_cv[:, :, 0], axis=1)

        distance_to_center = tf.concat([
           tf.expand_dims(tf.abs(y_min - im_center[:, 1]), axis=-1),
           tf.expand_dims(tf.abs(y_max - im_center[:, 1]), axis=-1),
           tf.expand_dims(tf.abs(x_min - im_center[:, 0]), axis=-1),
           tf.expand_dims(tf.abs(x_max - im_center[:, 0]), axis=-1),
        ], axis=-1)

        crop_width = 2.0 * self.lambda_mask_scaling * tf.reduce_max(distance_to_center, axis=-1)
        image_scaling = image_params["img_size"][0] / crop_width # TODO: this only works for same aspect ratio
        projection_matrix_crop_scaling = tf.concat([
            tf.expand_dims(image_params["img_size"][0] / crop_width, axis=-1),
            tf.expand_dims(image_params["img_size"][0] / crop_width, axis=-1),
            tf.ones_like(tf.expand_dims(crop_width, axis=-1)),
            tf.ones_like(tf.expand_dims(crop_width, axis=-1)),
        ], axis=-1)

        # changing sign of the z and y coz convention in Graphics is not the same in CV
        transformed_corners = tf.multiply(transformed_corners, [1, -1, -1])

        im_center_4x1 = tf.concat([
            tf.expand_dims((im_center[:, 0] - cam_mat[:, 0, 2]), axis=-1), # sings are flipped for x coz it xas different sign than z (see 2 lines above)
            tf.expand_dims(-(im_center[:, 1] - cam_mat[:, 1, 2]), axis=-1),
            tf.zeros_like(im_center[:, :2])
        ], axis=-1)

        # creating projection matrix
        projection_matrix_mul_cam_matrix_batch = tf.tile(tf.expand_dims(self.projection_matrix_mul_cam_matrix, axis=0),
                                                         [batch_size, 1, 1])
        projection_matrix_batch = tf.tile(tf.expand_dims(self.projection_matrix_add, axis=0),
                                          [batch_size, 1, 1])
        projection_matrix_mul_cam_matrix_batch = tf.multiply(projection_matrix_mul_cam_matrix_batch, cam_mat)
        projection_matrix_batch += tf.pad(projection_matrix_mul_cam_matrix_batch,
                                          [[0, 0], [0, 1], [0, 1]],
                                          "CONSTANT")

        projection_centering_vec = tf.expand_dims(tf.multiply(self.projection_centering_vec, im_center_4x1), axis=-1)
        projection_matrix_crop = tf.tile(tf.expand_dims(self.projection_centering_matrix_base, axis=0),
                                         [batch_size, 1, 1])
        projection_matrix_crop = tf.multiply(projection_matrix_crop, projection_centering_vec)

        projection_matrix_crop = tf.transpose(projection_matrix_crop + projection_matrix_batch, perm=[0, 2, 1])

        projection_matrix_zoom = tf.expand_dims(projection_matrix_crop_scaling, axis=1) * projection_matrix_crop

        transformed_corners = tf.concat([
            transformed_corners,
            tf.ones_like(transformed_corners[:, :, -1:])
        ], axis=-1)
        cube_vertices_clip = tf.matmul(transformed_corners, projection_matrix_zoom)

        pixels = dirt.rasterise_batch(
            vertices=cube_vertices_clip,
            faces=faces,
            vertex_colors=vertex_colors,
            background=tf.zeros([batch_size, image_params["img_size"][0], image_params["img_size"][1], 3]),
            width=image_params["img_size"][0], height=image_params["img_size"][1], channels=3
        )

        return pixels, cube_vertices_clip, im_center[:, :2], image_scaling


class MyDiffRendererStages(tf.keras.layers.Layer):
    def __init__(self, far_near_cut, model_files, init_rot, scaling, use_texture=False):
        super(MyDiffRendererStages, self).__init__()
        self.init_rot_quat = tf.cast(quaternion.from_rotation_matrix(init_rot), tf.float32)
        self.lambda_mask_scaling = 1.4

        self.vertices_all = []
        self.faces_all = []
        self.vertex_colors_all = []

        for it_obj in range(len(model_files)):

            if use_texture:
                vertices, faces, vertices_color = read_vertices_faces_textures(model_files[it_obj])
            else:
                vertices, faces = read_vertices_faces(model_files[it_obj])

            vertices *= scaling
            faces = tf.convert_to_tensor(faces, dtype=tf.int32)

            # init rotation of model
            vertices = tf.convert_to_tensor(vertices, dtype=tf.float32)
            vertices = quaternion.rotate(vertices, self.init_rot_quat)

            # Build the scene geometry, which is just an axis-aligned cube centred at the origin in world space
            # We replicate vertices that are shared, so normals are effectively per-face instead of smoothed
            vertices, faces = lighting.split_vertices_by_face(vertices, faces)

            min_vertices = tf.reduce_min(vertices, axis=-2)
            max_vertices = tf.reduce_max(vertices, axis=-2)

            # generate 3D object coordinates by setting vertex color to color corresponding to 3D bounding box
            if use_texture:
                vertex_colors = tf.cast(vertices_color, dtype=tf.float32) / 255.0
            else:
                vertex_colors = vertices - min_vertices  # origin to [0, 0, 0]
                vertex_colors = tf.div(vertex_colors, -min_vertices + max_vertices)  # scale by BB size

            self.vertices_all.append(tf.expand_dims(vertices, axis=0))
            self.faces_all.append(tf.expand_dims(faces, axis=0))
            self.vertex_colors_all.append(tf.expand_dims(vertex_colors, axis=0))

        self.vertices_all = tf.concat(self.vertices_all, axis=0)
        self.faces_all = tf.concat(self.faces_all, axis=0)
        self.vertex_colors_all = tf.concat(self.vertex_colors_all, axis=0)

        # Transform vertices from camera to clip space
        far = far_near_cut[0]
        near =  far_near_cut[1]

        projection_matrix_mul_cam_matrix = [
            [1. / (image_params["img_size"][0] / 2), 0., 0.],
            [0., 1. / (image_params["img_size"][1] / 2), 0.],
            [0., 0., 0.0]
        ]

        projection_matrix_add = [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ]

        self.projection_matrix_mul_cam_matrix = tf.convert_to_tensor(projection_matrix_mul_cam_matrix, dtype=tf.float32)
        self.projection_matrix_add = tf.convert_to_tensor(projection_matrix_add, dtype=tf.float32)

        projection_centering_vec = [1. / (image_params["img_size"][0] / 2), 1. / (image_params["img_size"][1] / 2), 0., 0.]
        self.projection_centering_vec = tf.convert_to_tensor(projection_centering_vec, dtype=tf.float32)

        self.projection_centering_matrix_base = tf.convert_to_tensor([
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ], dtype=tf.float32)


    def call(self, x):

        quat, pos, obj_ind, cam_mat, pos_init, quat_init = tf.split(x, (4, 3, 1, 9, 3, 4), axis=-1)
        batch_size = tf.shape(x)[0]
        obj_ind = tf.cast(obj_ind, dtype=tf.int32)
        cam_mat = tf.reshape(cam_mat, [-1, 3, 3])

        # get specific object vertices, faces, vertex colors (textures)
        vertices = tf.gather_nd(self.vertices_all, obj_ind)
        faces = tf.gather_nd(self.faces_all, obj_ind)
        vertex_colors = tf.gather_nd(self.vertex_colors_all, obj_ind)

        # computing image center, by computing projection of the center
        im_center = tf.matmul(tf.expand_dims(pos_init, axis=1), cam_mat, transpose_b=True)
        im_center = tf.reduce_sum(im_center, axis=1)
        im_center = tf.div(im_center, tf.expand_dims(im_center[:, -1], axis=1))

        # computing transformation of the vertices
        quat_broadcast = tf.expand_dims(quat, axis=-2)
        pos_broadcast = tf.expand_dims(pos, axis=-2)
        pos_broadcast_init = tf.expand_dims(pos_init, axis=-2)
        quat_broadcast_init = tf.expand_dims(quat_init, axis=-2)

        transformed_corners = quaternion.rotate(vertices, quat_broadcast) + pos_broadcast
        transformed_corners_init = quaternion.rotate(vertices, quat_broadcast_init) + pos_broadcast_init

        # computing scaling and crop size based on extreme projection points
        corners_projected_cv = tf.matmul(transformed_corners_init, cam_mat, transpose_b=True)
        corners_projected_cv = tf.div(corners_projected_cv, tf.expand_dims(corners_projected_cv[:, :, -1], axis=-1))

        y_min = tf.reduce_min(corners_projected_cv[:, :, 1], axis=1)
        y_max = tf.reduce_max(corners_projected_cv[:, :, 1], axis=1)
        x_min = tf.reduce_min(corners_projected_cv[:, :, 0], axis=1)
        x_max = tf.reduce_max(corners_projected_cv[:, :, 0], axis=1)

        distance_to_center = tf.concat([
           tf.expand_dims(tf.abs(y_min - im_center[:, 1]), axis=-1),
           tf.expand_dims(tf.abs(y_max - im_center[:, 1]), axis=-1),
           tf.expand_dims(tf.abs(x_min - im_center[:, 0]), axis=-1),
           tf.expand_dims(tf.abs(x_max - im_center[:, 0]), axis=-1),
        ], axis=-1)

        crop_width = 2.0 * self.lambda_mask_scaling * tf.reduce_max(distance_to_center, axis=-1)
        image_scaling = image_params["img_size"][0] / crop_width # TODO: this only works for same aspect ratio
        projection_matrix_crop_scaling = tf.concat([
            tf.expand_dims(image_params["img_size"][0] / crop_width, axis=-1),
            tf.expand_dims(image_params["img_size"][0] / crop_width, axis=-1),
            tf.ones_like(tf.expand_dims(crop_width, axis=-1)),
            tf.ones_like(tf.expand_dims(crop_width, axis=-1)),
        ], axis=-1)

        # changing sign of the z and y coz convention in Graphics is not the same in CV
        transformed_corners = tf.multiply(transformed_corners, [1, -1, -1])

        im_center_4x1 = tf.concat([
            tf.expand_dims((im_center[:, 0] - cam_mat[:, 0, 2]), axis=-1), # sings are flipped for x coz it xas different sign than z (see 2 lines above)
            tf.expand_dims(-(im_center[:, 1] - cam_mat[:, 1, 2]), axis=-1),
            tf.zeros_like(im_center[:, :2])
        ], axis=-1)

        # creating projection matrix
        projection_matrix_mul_cam_matrix_batch = tf.tile(tf.expand_dims(self.projection_matrix_mul_cam_matrix, axis=0),
                                                         [batch_size, 1, 1])
        projection_matrix_batch = tf.tile(tf.expand_dims(self.projection_matrix_add, axis=0),
                                          [batch_size, 1, 1])
        projection_matrix_mul_cam_matrix_batch = tf.multiply(projection_matrix_mul_cam_matrix_batch, cam_mat)
        projection_matrix_batch += tf.pad(projection_matrix_mul_cam_matrix_batch,
                                          [[0, 0], [0, 1], [0, 1]],
                                          "CONSTANT")

        projection_centering_vec = tf.expand_dims(tf.multiply(self.projection_centering_vec, im_center_4x1), axis=-1)
        projection_matrix_crop = tf.tile(tf.expand_dims(self.projection_centering_matrix_base, axis=0),
                                         [batch_size, 1, 1])
        projection_matrix_crop = tf.multiply(projection_matrix_crop, projection_centering_vec)

        projection_matrix_crop = tf.transpose(projection_matrix_crop + projection_matrix_batch, perm=[0, 2, 1])

        projection_matrix_zoom = tf.expand_dims(projection_matrix_crop_scaling, axis=1) * projection_matrix_crop

        transformed_corners = tf.concat([
            transformed_corners,
            tf.ones_like(transformed_corners[:, :, -1:])
        ], axis=-1)
        cube_vertices_clip = tf.matmul(transformed_corners, projection_matrix_zoom)

        pixels = dirt.rasterise_batch(
            vertices=cube_vertices_clip,
            faces=faces,
            vertex_colors=vertex_colors,
            background=tf.zeros([batch_size, image_params["img_size"][0], image_params["img_size"][1], 3]),
            width=image_params["img_size"][0], height=image_params["img_size"][1], channels=3
        )

        return pixels, cube_vertices_clip, im_center[:, :2], image_scaling