# GBI RANODE

This part of the code corresponds to the RANODE section in the paper ["Generator Based Inference (GBI)"](https://arxiv.org/abs/2405.08889) by Chi Lung Cheng, Ranit Das, Runze Li, Radha Mastandrea, Vinicius Mikuni, Benjamin Nachman, David Shih and Gup Singh.

## Installation

The environment requirement for RANODE based inference is available in `requirements.txt`, it can be installed by running:

```
conda env create -f environment.yml --prefix /path/gbi_ranode_env
```

To setup the rest of environment variables, run

```
source setup.sh
```

During first executation, user will be prompted to enter the input and output directory. The input directory should contain the files listed in the Dataset section.

## Dataset

- Simulated QCD background from official LHCO dataset: https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5
- Extra simulated QCD background : https://zenodo.org/records/8370758/files/events_anomalydetection_qcd_extra_inneronly_features.h5
- Extended parametric W->X(qq)Y(qq) signal : https://zenodo.org/records/15384386/files/events_anomalydetection_extended_Z_XY_qq_parametric.h5
- Signal ensembles with trainvaltest splitting : lumi_matched_train_val_test_split_signal_features_W_qq.h5


### Tutorials

["Luigi Analysis Workflow (LAW)"](https://github.com/riga/law) is used to construct this project. First, one needs to setup the law task list by running
```
conda activate /path/gbi_ranode_env
source setup.sh

law index
```

After this different tasks can be run with law by commands like:

```
law taskname --version output_postfix --flags XXX
```

### Command Line Interface

```
law ScanOverTrueMu --version test_0 --mx 100 --my 500 --ensemble 1 --workers 3
```


