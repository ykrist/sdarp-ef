# Extended Fragments Algorithm for the Selective Dial-A-Ride Problem
This repo contains the implementation of the algorithm presented in [this paper](https://doi.org/10.1016/j.cor.2021.105649).

# Installation

## Cloning
The repo contains sub-modules, so clone with 
```bash
git clone --recurse-submodules <URL>
```

## Requirements
To run the code, you will need to have the [Rust toolchain](https://rustup.rs/) installed which provides the `cargo` shell command.  You will also need Conda (`conda` command).  It can be found [here](https://docs.conda.io/en/latest/miniconda.html).

## Installer
Run the installer with
```
./create_environment.sh
```
which should create the necessary conda environment, extract the problem instances and compile the code.

# Running
Before running anything make sure the correct Conda environment is selected:
```bash
conda activate ./env
```
There are two script files: `data_indices.py` and `sdarp_ef.py`.

The main script is `sdarp_ef.py` and requires one mandatory argument, which is an integer index into the problem set. The available instances and their corresponding index can be viewed with the other script:
```bash
python data_indices.py -m all
0 30N_4K_A
1 30N_5K_A
2 40N_4K_A
3 40N_5K_A
4 44N_4K_A
5 44N_5K_A
6 50N_4K_A
7 50N_5K_A
8 60N_4K_A
9 60N_5K_A
...
```

So to solve instance `30N_4K_A`, run
```bash
python -O sdarp_ef.py 0
```
Note the `-O` flag to Python, which skips debug assertions. Performance will suffer if this flag is not supplied.

The main script will create log directories under `./logs/` in which output can be found.

The parameter settings from the paper can be found in the `parameters/` directory and can be loaded using
the `--load-params` flag in the main script.

Further documentation can be found using the `--help` flags on both scripts.

The instance generator used in the paper is provided as well (`instance_generator.py`).