CXX = g++
MPI_CXX = mpic++

all: seriale openmp mpi

seriale: seriale.cpp
	$(CXX) -o seriale seriale.cpp

openmp: openmp.cpp
	$(CXX) -o openmp openmp.cpp

mpi: mpi.cpp
	$(MPI_CXX) -o mpi mpi.cpp

clean:
	rm -f seriale openmp mpi
