# AMSCProject
Implementation of a tiny neural network from scratch in C++ and training it in parallel using MPI and OpenMP. Exam of "Advanced Methods for Scientific Computing".


In this project I tried to build a little feed forward neural network which simulates the behaviour of the truth table of the XOR logical operation.
So I built the neural network and then a training function with the stochastic gradient descent inside, in the most basilar version with a mini batch concerning the whole dataset, so without shuffling data. In our case "data" just means the input of the truth table (following):

inputs            outputs
0 0               0
0 1               1
1 0               1
1 1               0

As you can see in the repository there are 3 different .cpp files:
seriale.cpp   openmp.cpp  mpi.cpp
and 3 different training header files:
training.hpp  trainingomp.hpp trainingmpi.cpp
Running the serial, the MPI or the OpenMP version will respectively call the serial, MPI or OpenMP training function. All of them recall previously this headers in order:
defclassi.hpp funzioni.hpp implementazioneclassi.hpp
defclassi.hpp contains the definition of all the class of the neural network, funzioni.hpp contains some useful global functions, implementazioneclassi.hpp has inside the real construction of classes present in defclassi. The parallel versions mainly aimed to demonstrate that parallelizing the training process gives a gain in terms of time.
Unfortunately, despite almost everything works as expected, the training phase does not bring to an acceptable approximation of the real XOR truth table.
