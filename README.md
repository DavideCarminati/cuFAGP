# Parallel Gaussian process with kernel approximation in CUDA

CUDA implementation of Gaussian process with approximated kernel. The paper liked to this repo is available as a pre-print [here](https://arxiv.org/abs/2403.12797). The mathematical formulation was originally presented in V. Joukov and D. Kulić (2022) (pre-print available [here](https://arxiv.org/abs/2008.09848)).
### Requirements
* `gcc` v9.4.0
* [`CUDA` toolkit v12.2](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
* `CMake` >= v3.18
* [Eigen3 v3.4](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* `Boost` >= v1.71
* Matlab/Octave
The code is tested in Ubuntu 20.04.6 LTS. Figures are generated in Matlab R2022a. Octave 5.2.0 is also supported.
### Building the executables
In the desired folder, clone this repository:
```sh
git clone git@github.com:DavideCarminati/cuFAGP.git
```
Then `cd` in the folder and build the executables with CMake:
```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
Executables are located in the `build` folder.
### Run the example script
Once executables are built, run in the terminal:
```sh
cd ..
source example.sh
```
to replicate the results in the paper.

### How to cite
```
@misc{carminati2024parallel,
      title={Parallel Gaussian process with kernel approximation in CUDA}, 
      author={Davide Carminati},
      year={2024},
      eprint={2403.12797},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```