import os
import glob

from plyfile import PlyData
from utils.pose_error_metrics import *
from test_scripts.test_linemod_all import eval_checkpoint


scaling_2_mm = 1000.0

objects_linemod_list = ['ape', 'benchvise', 'cam', 'can',
                        'cat', 'driller', 'duck', 'eggbox',
                        'glue', 'holepuncher', 'iron', 'lamp', 'phone']

objects_occ_linemod_list = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']

diameters = [102.09865663, 247.50624233, 172.49224865, 201.40358597,
             154.54551808, 261.47178102, 108.99920102, 164.62758848,
             175.88933422, 145.54287471, 278.07811733, 282.60129399, 212.35825148]

diameters = {
    'ape': 102.09865663,
    'benchvise': 247.50624233,
    'cam': 172.49224865,
    'can': 201.40358597,
    'cat': 154.54551808,
    'driller': 261.47178102,
    'duck': 108.99920102,
    'eggbox': 164.62758848,
    'glue': 175.88933422,
    'holepuncher': 145.54287471,
    'iron': 278.07811733,
    'lamp': 282.60129399,
    'phone': 212.35825148,
}

def eval_checkpoints(weights_file):


        model_name = weights_file.split('/')[-2]

        out_folder = os.path.join(config.results_folder, model_name)

        eval_options = {
            "out_folder": out_folder,
            "ckpt_path": weights_file,
            "dataset_folder": config.dataset_folder,
            "mesh_folder": config.mesh_folder,
            "eval_occlusion": config.eval_occlusion,
        }

        eval_checkpoint(predict_eval_and_save_external, eval_options)


def read_data_my_approach_separate_objects(object_it, predict_pose_file):
    predict_data_all = []
    gt_data_all = []

    predict_pose_file_object = os.path.join(predict_pose_file, object_it)
    if os.path.exists(predict_pose_file_object):
        file_names = sorted(glob.glob(os.path.join(predict_pose_file_object, '*.npy')))
        for it_file in file_names:
            predict_data = np.load(it_file, allow_pickle='TRUE').item()
            predict_data_all.append(predict_data["pose_pred"])
            gt_data_all.append(predict_data["pose_gt"])

    return predict_data_all, gt_data_all


def read_eval_poionts(cls_name):

    model_folder = config.mesh_folder
    ply_file = os.path.join(model_folder, cls_name + '_4000.ply')

    with open(ply_file, 'rb') as f:
        mesh = PlyData.read(f)
        vertices_model_x = np.expand_dims(mesh['vertex']['x'], axis=-1)
        vertices_model_y = np.expand_dims(mesh['vertex']['y'], axis=-1)
        vertices_model_z = np.expand_dims(mesh['vertex']['z'], axis=-1)

    points = np.concatenate((vertices_model_x, vertices_model_y, vertices_model_z), axis=-1) * scaling_2_mm

    return points


def eval_6D_pose_rot_mat_auc(pos_est, rot_mat_est, eval_dict, metrics, treshold_dict):

    eval_pts = eval_dict["eval_pts"]
    R_gt = eval_dict["R_gt"]
    pos_gt = eval_dict["pos_gt"]

    out_dict = {}

    if metrics["add"]:
        add_error = add(rot_mat_est, pos_est, R_gt, pos_gt, eval_pts)
        auc_add = add_error <= treshold_dict["add"]

        out_dict["add_auc"] = auc_add
    else:
        out_dict["add_auc"] = 0

    if metrics["adi"]:
        adi_error = adi(rot_mat_est, pos_est, R_gt, pos_gt, eval_pts)
        auc_adi = adi_error <= treshold_dict["adi"]

        out_dict["adi_auc"] = auc_adi
    else:
        out_dict["adi_auc"] = 0

    return out_dict



def compute_add_score(pred_pose, gt_pose, diameter, object_name, compute_adds_error):
    threshold = {
        "add": np.linspace(0, 100, 10 * 100),
        "adi": np.linspace(0, 100, 10 * 100),
    }

    if compute_adds_error:
        metrics_to_use = {
            "add": False,
            "adi": True,
        }
    else:
        metrics_to_use = {
            "add": True,
            "adi": False,
        }

    eval_pts = read_eval_poionts(object_name)

    options = {
        "eval_pts": eval_pts,
        "metrics_to_use": metrics_to_use,
        "threshold": threshold,
    }

    samples_num = len(pred_pose)

    all_add = np.zeros((samples_num, np.shape(options["threshold"]["add"])[0]))
    all_adi = np.zeros((samples_num, np.shape(options["threshold"]["adi"])[0]))

    for it_pose in range(samples_num):

        R_gt = gt_pose[it_pose][:3, :3]
        pos_gt = gt_pose[it_pose][:3, 3]

        pred_pose_it = pred_pose[it_pose][:3, 3]
        pred_R_it = pred_pose[it_pose][:3, :3]

        eval_dict = {
            "eval_pts": options["eval_pts"],
            "R_gt": R_gt,
            "pos_gt": pos_gt,
        }

        out_dict = eval_6D_pose_rot_mat_auc(pred_pose_it, pred_R_it, eval_dict,
                                            options["metrics_to_use"], options["threshold"])

        all_add[it_pose, :] = out_dict["add_auc"]
        all_adi[it_pose, :] = out_dict["adi_auc"]

    if compute_adds_error:
        add_percent = np.sum(all_adi, axis=0) / samples_num
    else:
        add_percent = np.sum(all_add, axis=0) / samples_num

    d_01 = 0.1 * diameter
    d_01_ind = int(d_01 * 10.0)

    return add_percent[d_01_ind]


def compute_add(results_folder):

    if config.eval_occlusion:
        objects_list = objects_occ_linemod_list
    else:
        objects_list = objects_linemod_list

    add_scores = np.zeros(len(objects_list))

    for object_it in range(len(objects_list)):

        if (objects_list[object_it] == "glue" or objects_list[object_it] == "eggbox"):
            compute_adds_error = True
        else:
            compute_adds_error = False

        pred_pose, gt_poses = read_data_my_approach_separate_objects(objects_list[object_it], results_folder)
        score = compute_add_score(pred_pose, gt_poses, diameters[objects_list[object_it]], objects_list[object_it], compute_adds_error)
        add_scores[object_it] = score

    return add_scores

def compute_all_scores_from_config():

    model_name = config.weights_file.split('/')[-2]
    results_folder = os.path.join(config.results_folder, model_name)

    add_scores = compute_add(results_folder)

    print(add_scores)

if __name__ == '__main__':

    model_to_eval = "linemod_single"  # options: linemod_single, linemod_stages , linemod_occlusion_single, linemod_occlusion_stages

    if model_to_eval == "linemod_single":
        from eval_scripts.configs_eval import config_linemod_single as config
    elif model_to_eval == "linemod_stages":
        from eval_scripts.configs_eval import config_linemod_stages as config
    elif model_to_eval == "linemod_occlusion_single":
        from eval_scripts.configs_eval import config_linemod_occlusion_single as config
    elif model_to_eval == "linemod_occlusion_stages":
        from eval_scripts.configs_eval import config_linemod_occlusion_stages as config

    if config.model == "single":
        from models.u_dense_net_attention import predict_eval_and_save_external
    elif config.model == "stages":
        from models.u_dense_net_attention_stages import predict_eval_and_save_external

    eval_checkpoints(config.weights_file)
    compute_all_scores_from_config()