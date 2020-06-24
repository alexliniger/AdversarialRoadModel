# Discriminating Kernel Algorithm
This is a C++ implementation of the discriminating kernel algorithm used to compute the safe set using the idea of an adversarial road as proposed in "Safe Motion Planning for Autonomous Driving using an Adversarial Road Model".

## Installation 

To install all the dependencies run
```
bash install.sh
```
this clones `nlohmann/json`, and `eigen`, from their git repo, and saves them in a folder External. 

Once all dependencies are installed `cmake` can be used to build the project
```
cmake CMakeLists.txt
make
```
To run the code simply execute the `DiscKernelAlgo`
```
./DiscKernelAlgo
```

## Running Code

### Change Car Parameters

The parameters of the car can be easily changed in `car.json`, including the maximum curvature `kappa_max`.

### Chang Grid Parameters

The number of grid points and the sampling time can be changed `types.h`. Note that the sampling time and the grid resolution are generally coupled, if the sampling time is too small relative to the grid spacing the results are not representative.

### Generating Training Set

If the goal is to generate the training set used in the paper the script `run.sh` can be executed which automatically computes the discriminating kernel for all the maximum curvatures.
