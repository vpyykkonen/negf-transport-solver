# negf-transport-solver
Simulation of transport through (two-terminal) transport setups with a tight-binding scattering system.

The simulated system consists of two reservoirs and a scattering system, which is connected to the reservoirs via leads. The scattering system and the leads are considered as tight-binding models. The leads are semi-infinite chains while the scattering system tight-binding geometry can be specified freely. The leads can be either normal or superconductors with specified order parameters. The scattering system is modeled as an on-site Hubbard model at the mean-field level. The mean fields are determined self-conistently with the non-equilibrium dynamics. 

The system can be driven out of equilibrium by setting different chemical potentials, temperatures or superconducting-order-parameter phases for the two leads. The observables such as current and particle number are obtained for the scattering system assuming that the system has reached a steady state. In other words, the transient effects and initial correlations are neglected. The observable expectation values and the self-consistent fields are determined by non-equilibrium Green's function method.


## Installation
Clone the source code

Install Eigen and hdf5 and add the appropriate links to the Makefile.
Eigen version 3.3.9 has been used in the code but probably newer versions also work. For Eigen, just download the software and link the location similarly to the example in the Makefile. For hdf5 you can use the standard libhdf5-dev package if its installed. Alternatively, you can use e.g. Anaconda package handling to install hdf5 and link the anaconda libraries similarly to the given examples. Note in modifying the Makefile that specific order of the `-I`, `-rpath` and `-L` are important. 

Use the command 'make all' to compile the code. Optionally, you can add the handle -jN to parallelize the compilation, eg. 'make all -j4' to use 4 threads.

## Usage
The code is used in two stages. First, the analysis is prepared by specifying the scattering system geometry and two-terminal setup parameters in parameters.cfg and geometry.cfg files.
The analysis is prepared by command `./prepare_analysis ./parameters.cfg ./geometry.cfg`, which prepares the data point files. Each data point corresponds to a hdf5 file containing the parameter. These files can then by processed by the `process_datapoint` program by `./process_datapoints path_to_data.h5 scf_params.cfg`, where `scf_params.cfg` specifies the self-consistent field algorithms.  The program will also generate a bash file to process the data point files and a sbatch file for cluster processing.


## Structure of the code
The data point preparation for analysis is described in the the main file `prepare_datapoints.cpp` and data point processing in the main file `process_datapoint.cpp`. The configuration files for the details are read by routines defined in `config_parser.cpp` and `file_io.cpp` files.

The parts of the two-terminal setup correspond to classes Lead and ScatteringSystem specified in Lead.h and ScatteringSystem.h. The two-terminal setup corresponds to classes TwoTerminalSetup and SSJunction. The former is used in most situations but for practical reasons the SSJunction is used in non-equilibrium situations when both leads are superconducting.


The self-consistent fields are solved by fixed-point algorithms specified in ScfSolver and ScfMethod classes, where the ScfSolver is the general solver and ScfMethods are specific algorithms for the purpose.

The program solves the frequency-dependent Green's functions and performs an inverse Fourier transform to obtain the time-dependent functions. The inverse Fourier transform is performed with a doubly adaptive quadrature algorithm for obtaining the integral. In the Green's functions, the essential information is localized around specific regions of frequency not known a priori, which renders standard non-adaptive trapezoid approach inpractical. In contrast the double-adaptive algorithm by Terje Espelid detects the important variable region and focuses the function evaluation to these regions. The algorithm is specified in the files da2glob.cpp and da2glob.h and it uses the classes Interval and Global to handle single intervals in the integration grid and their collection, respectively. The quadrature rules are stored in the quadrature.cpp and quadrature.h files. The weights of the used Newton-Cotes rules and the corresponding null rule weights for error estimation are stored in `da2glob_weights` folder.

The equilibrium Green's functions at finite temperature are often obtained via the method of Matsubara summation. It is based on the fact that instead of the continuous frequency, the discrete imaginary frequency contains the essential information nearby the origin of the complex frequency plane. However, the summation in its standard form converges slowly with respect to the number of elements. We have implemented Pade summation algorithm which provides quadratic convergence instead of linear one. The frequencies are obtained by the functions in `pade_frequencies.cpp`.

