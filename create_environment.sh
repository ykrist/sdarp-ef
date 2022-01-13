#!/bin/bash -i

function check_cmd_exists {
    CMD_PATH=`which $1`
    if [[ $? -eq 0 ]] ; then 
        echo "$1 found: $CMD_PATH"
        unset CMD_PATH
        return
    else 
        echo "error: $1 is not installed or cannot be found on PATH" >&2
        exit 1
    fi
}

check_cmd_exists conda
check_cmd_exists cargo

set -e

CONDA_ENV_PATH=./env
CONDA_ENV_PATH_FULL=`readlink -f ./env`

if [[ -d env/ ]] ; then 
    echo "Existing environment found"
else
    echo "Creating conda env (this may take a few minutes)..."
    conda env create -f env.yml -p $CONDA_ENV_PATH
fi

echo "Setting environment variables..."
conda env config vars set -p $CONDA_ENV_PATH HDF5_DIR=$CONDA_ENV_PATH_FULL LD_LIBRARY_PATH=${CONDA_ENV_PATH_FULL}/lib:${LD_LIBRARY_PATH} DATA_ROOT=`readlink -f ./data`
conda env config vars list -p ./env


echo "Adding Python modules..."
mkdir -p lib/fraglib-py/python
conda activate $CONDA_ENV_PATH
conda develop lib/fraglib-py/python lib/oru lib/phd-utils
cp config.yaml lib/phd-utils/


(
    cd data
    echo "Extracting instances..."
    for F in compressed/* ; do 
        tar -Jxf $F
    done    
)

(
    cd lib/fraglib-py 
    echo "Compiling Rust extensions for Python..."
    ./build.sh
)

echo "Done."
echo "Use \"conda activate ${CONDA_ENV_PATH}\" to activate the environment"