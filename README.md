# Spatial Attention Improves Iterative 6D Object Pose Estimation

The code accompanying our 3DV 2020 publication.

- Authors: [Stefan Stevsic](https://ait.ethz.ch/people/stevsics/), [Otmar Hilliges](https://ait.ethz.ch/people/hilliges/)
- Project page: https://ait.ethz.ch/projects/2020/attention6dpose/


## Installation

To run the code you will need the following python packages:

- glob
- ply
- numpy
- purr
- scipy
- tensorflow 1.14.0
- tensorflow_graphics 1.0.0
- dirt 0.3.0 (installation instructions: https://github.com/pmh47/dirt)

We tested the code on Ubuntu 16.04 in the python 3.7 environments.


## Preparing the data 

The main results in the paper use the [Pvnet](https://github.com/zju3dv/clean-pvnet) predictions for initial poses. Thus, the first step is to obtain these poses. The Pvnet project page (link: https://github.com/zju3dv/clean-pvnet) provides the [Linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EXK2K0B-QrNPi8MYLDFHdB8BQm9cWTxRGV9dQgauczkVYQ?e=beftUz) and [linemod occlusion](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/ESXrP0zskd5IvvuvG3TXD-4BMgbDrHZ_bevurBrAcKE5Dg?e=r0EgoA) datasets as well. Download these datasets and place them in the 'src/resources/pvnet_data/LINEMOD' and 'src/resources/pvnet_data/OCCLUSION_LINEMOD' folders. 

To obtain the initial poses follow the instructions at the [Pvnet](https://github.com/zju3dv/clean-pvnet) project page. Place the Pvnet results of the Linemod and Linemod Occlusion datasets in the 'src/resources/pvnet_data/init_poses/linemod' and 'src/resources/pvnet_data/init_poses/occlusion' folders respectively. To obtain the initial poses, we use the standard Pvnet output, which can be obtained by running the instructions described at Testing, step 3. at the [Pvnet](https://github.com/zju3dv/clean-pvnet) project page.

Once you have the results, you need to generate the tfrecord files that are used to evaluate our network. Navigate to the 'src/preprocessing_scripts/' and run the 'generate_test_data.py' python script. The script will generate the tfrecord files and place them in the 'src/resources/datasets/' folder.

## Model weighgts

The model weights used for the final results in the paper are available [here](). Download the weights, unzip them and place them in the 'src/weights/' folder.

## Testing 

We only provide evaluation code for our model. To evaluate the model navigate the 'src/eval_scripts/ ' folder and run the 'eval_linemod.py' python script. In order to evaluate different models modify the value of the 'model_to_eval' variable to one of the following options: linemod_single, linemod_stages , linemod_occlusion_single, linemod_occlusion_stages. Using the options containing 'single' results in the evaluation of the single-stage variant of our model, while using the ones containing 'stages' results in the evaluation of the multi-stage model variant.

## Citation

If using this code-base in your research, please cite the following publication:

'''

@inproceedings{stevsic20203dv,
  title={Spatial Attention Improves Iterative 6D Object Pose Estimation},
  author={{Stevšić}, Stefan and Hilliges, Otmar},
  booktitle={2020 International Conference on 3D Vision (3DV)},
  year={2020},
  organization={IEEE}
}

'''

