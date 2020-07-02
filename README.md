# Long-Horizon Visual Planning with Goal-Conditioned Hierarchical Predictors
#### [[Project Website]](https://orybkin.github.io/video-gcp/)

[Karl Pertsch](https://kpertsch.github.io/)<sup>*1</sup>, [Oleh Rybkin](https://www.seas.upenn.edu/~oleh/)<sup>*2</sup>, 
[Frederik Ebert](https://febert.github.io/)<sup>3</sup>, [Chelsea Finn](http://people.eecs.berkeley.edu/~cbfinn/)<sup>4</sup>,
[Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/)<sup>2</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>3</sup><br/>
(&#42; equal contribution)

<sup>1</sup>University of Southern California 
<sup>2</sup>University of Pennsylvania 
<sup>3</sup>UC Berkeley 
<sup>4</sup>Stanford 

<a href="https://orybkin.github.io/video-gcp/">
<p align="center">
<img src="https://orybkin.github.io/video-gcp/resources/teaser.gif" width="800">
</p>
</img></a>

This is the official Pytorch implementation for our paper **Long-Horizon Visual Planning with Goal-Conditioned Hierarchical Predictors**.


## Installation

To install the module, run the following commands:

```
git clone --recursive git@github.com:orybkin/video-gcp.git
cd video-gcp

virtualenv -p $(which python3) ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
python3 setup.py develop
```

## Model Training

To train a tree-structured prediction model start by setting paths for data and experiment logs:
```
export GCP_DATA_DIR=./data
export GCP_EXP_DIR=./experiment_logs
``` 
Note that the data will be automatically downloaded upon first running the model training code.  

To start training the GCP model on the 25-room navigation dataset, run: 
```
cd gcp
python3 prediction/train.py --path=../experiments/prediction/25room/gcp_tree/ --skip_first_val=1
```
The training logs get written into `GCP_EXP_DIR` and can be displayed by opening a Tensorboard in this folder:
`tensorboard --logdir=$GCP_EXP_DIR`.

## Planning Experiments
For running planning experiments we need to set up a virtual display for rendering (skip this step if you are not running on a headless machine).
```
Xvfb -screen 0 320x240x24 &
export DISPLAY=:0
```

Next, we need to install the `gym-miniworld` submodule:
```
cd gym-miniworld
python3 setup.py develop
```

To run a planning evaluation run with the previously trained model, execute:
```
cd gcp
python3 planning/run.py ../experiments/control/25room/gcp_tree/mod_hyper.py
```

To compute the control performance metrics (success rate and trajectory cost), run:
```
python evaluation/compute_control_perf.py --path=${GCP_EXP_DIR}/control/25room/gcp_tree/scores_0to99.pkl --n_rooms=25
```


## Modifying the Code
### Running on a new dataset
If you want to run our model on a new dataset, the easiest is to subclass [```FolderSplitVarLenVideoDataset```](gcp/datasets/data_loader.py)
 and overwrite the ```__getitem__()``` method to load your data. Then just point to your custom 
data loader in the config file. Our code expects the data loader's ```__getitem__()``` function to output a dictionary with
the following structure:
```
dict({
    'traj_seq': (time, channels, heigth, width) # zero-padded sequence of max_seq_len
    'pad_mask': (time,)                         # one for all time steps that are zero-padded, 0 otherwise
    'I_0': (channels, height, width)            # start image
    'I_g': (channels, height, width)            # goal image
    'end_ind': int                              # index of goal image in the demo sequence
    'states': (time, state_dim)                 # (optional) state sequence corresponding to image sequence, for training state regressor
    'actions': (time, action_dim)               # (optional) action sequence for training inverse model
})
```
Further your dataset folder is expected to have the following structure:
```
< dataset_path >
       |-- dataset_spec.py      # specifies state/action dimensions + sequence length
       |-- train/               # training trajectory files
       |-- val/                 # validation trajectory files
       |-- test/                # test trajectory files
```
You can find an example ```dataset_spec.py``` file  [here](experiments/data_gen/misc/example_dataset_spec.py). 
By default the data loader will search for all ```.h5``` files in the ```train/val/test``` subfolders to compile the list of filenames. 
If you want to load from a different file format or have a different folder structure you need to adjust the ```get_filenames()``` function accordingly.

### Modifying the predictive model
The proposed goal-conditioned predictors are implemented in the [```prediction```](gcp/prediction) directory.
Below we describe the functionality implemented in the different files to allow for quick adjustments.

|File        | Description         |
|:------------- |:-------------|
| [```train.py```](gcp/prediction/train.py) | Tranining script. |
| [```sequential.py```](gcp/prediction/sequential.py) | A naive sequential goal-conditioned predictor (GCP). This works well on some data, but fails to scale to long horizons. |
| [```blox/torch/models/vrnn.py```](blox/torch/models/vrnn.py) | The variational RNN, which is the basis of the sequential GCP. |
| [```tree/tree.py```](gcp/prediction/tree/tree.py) | The proposed hierarchical GCP which scales to long horizons. |
| [```tree/tree_module.py```](gcp/prediction/tree/tree_module.py) | The basic module for hierarchical prediction: predicting a state between two other states. |
| [```tree/tree_lstm.py```](gcp/prediction/tree/tree_lstm.py) | Tree LSTM implementation. |
| [```blox/torch/encoder_decoder.py```](blox/torch/encoder_decoder.py) | The encoder and decoder networks and various decoding distributions. |
| [```adaptive_binding```](gcp/prediction/adaptive_binding) | The hierarchical model wih adaptive binding. |

A significant amount of utils developed by us for this and other projects are in the [blox](https://github.com/orybkin/blox-nn) repo,
including tools for quick fully connected, convolutional, and recurrent architectures, operations on nested tensor dictionaries,
tools for probabilistic deep learning, and tools to interface numpy and torch code.

### Modifying the planning algorithm
The code that implements the image-based planning and control approach described in the paper is in the [```planning```](gcp/planning) directory. 
Below we describe the functionality implemented in the different files to allow for quick adjustments.

|File        | Description         |
|:------------- |:-------------|
| [```run.py```](gcp/planning/run.py) | Launch script for planning & control experiments. |
| [```tree_optimizer.py```](gcp/planning/tree_optimizer.py) | Core hierarchical planning algorithm for GCP-tree latent optimization. |
| [```planner_policy.py```](gcp/planning/planner_policy.py) | Policy wrapper that calls planner and follows plan using inverse model. |
| [```cem/cem_planner.py```](gcp/planning/cem/cem_planner.py) | CEM planners for flat and hierarchical prediction models. |
| [```cem/cem_simulator.py```](gcp/planning/cem/cem_simulator.py) | Wrapper around predictive model for interface with CEM planner. |
| [```cem/sampler.py```](gcp/planning/cem/sampler.py) | Samples latents / actions for CEM planning, calls latent optimizer. |
| [```cem/cost_fcn.py```](gcp/planning/cem/cost_fcn.py) | Cost functions for CEM planning. |

Most of the infrastructure for the planning experiments (start / goal loading, logging, interaction between environment and agent)
directly builds on the [Visual Foresight codebase](https://github.com/SudeepDasari/visual_foresight) and can be found in the 
[```infra```](gcp/planning/infra) directory.

### Generating navigation data with new layout
We provide an example script for using a PRM planner to generate a new navigation dataset in an environment with 16 rooms.
The layout can be further adjusted [here](gcp/planning/infra/envs/miniworld_env/utils/multiroom2d_layout.py). 
The ```nworkers``` argument allows for parallelized data generation.
```
cd recursive_planning
python3 planning/run.py ../experiments/data_gen/nav_16rooms/mod_hyper.py --nworkers=4
```

## Downloading the datasets
The training code automatically downloads the required datasets if they are not already in the expected folders. 
However, if you want to download the datasets independently, you can find the zip files here:

|Dataset        | Link         | Size |
|:------------- |:-------------|:-----|
| 9-room navigation | [https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_9rooms.zip](https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_9rooms.zip) | 140MB |
| 25-room navigation |[https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_25rooms.zip](https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_25rooms.zip)| 395MB|
| Sawyer |[https://www.seas.upenn.edu/~oleh/datasets/gcp/sawyer.zip](https://www.seas.upenn.edu/~oleh/datasets/gcp/sawyer.zip)|395MB|
| Human3.6M 500-frame | [https://www.seas.upenn.edu/~oleh/datasets/gcp/h36m.zip](https://www.seas.upenn.edu/~oleh/datasets/gcp/h36m.zip) | 14GB|


## Bibtex
If you find this work useful in your research, please consider citing:
```
@article{pertsch2020gcp,
    title={Long-Horizon Visual Planning with Goal-Conditioned Hierarchical Predictors},
    author={Karl Pertsch and Oleh Rybkin and Frederik Ebert 
    and Chelsea Finn and Dinesh Jayaraman and Sergey Levine},
    year={2020},
    journal={arXiv preprint arXiv:2006.13205},
}
```

## Acknowledgements
Parts of the planning code are based on the [Visual Foresight codebase](https://github.com/SudeepDasari/visual_foresight).

In parallel with this method, we developed the [blox](https://github.com/orybkin/blox-nn) python module, which contains
many useful utilities and architectures for pytorch. Check it out and let us know if it was useful!
