#!/bin/bash -i
CONDA_ENV_NAME=sdarp

set -e
ENV_EXISTS=$(conda info --envs -q | grep -E 'sdarp[[:space:]]+' -q && echo 1 || echo 0)

if [[ $ENV_EXISTS -eq 0 ]] ; then 
    echo "Creating conda env..."
    conda env create -f env.yml
    conda activate $CONDA_ENV_NAME

    conda env config vars set HDF5_DIR=$CONDA_PREFIX
    conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
    conda env config vars set DATA_ROOT=`readlink -f ./data`
    conda deactivate
else
    echo "Existing environment found (${CONDA_ENV_NAME})"
fi

conda activate $CONDA_ENV_NAME

mkdir -p lib/fraglib-py/python

conda develop lib/fraglib-py/python lib/oru lib/phd-utils

cp config.yaml lib/phd-utils/

(
    cd data
    echo "Extracting instances..."
    for F in compressed/* ; do 
        tar -Jxf $F
    done    
)

cd lib/fraglib-py 
echo "Compiling Rust extensions for Python..."
exec ./build.sh
