#!/usr/bin/env bash

action() {

    # set python path in current directory
    this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export PYTHONPATH="${this_dir}:${PYTHONPATH}"

    export LAW_HOME="${this_dir}/.law"
    export LAW_CONFIG_FILE="${this_dir}/law.cfg"

    module load python

    conda activate /global/common/software/m3246/HAXAD/software/haxad3

    echo $PYTHONPATH

    # define output directory
    export OUTPUT_DIR="/pscratch/sd/m/mukyu/SBI_RANODE"

    # add submodules to the path
    export RANODE="${this_dir}/submodules/RANODE/src"

}

action