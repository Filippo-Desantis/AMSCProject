#ifndef DEFCLASSI_HPP
#define DEFCLASSI_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <mpi.h>

//definizione delle classi, partendo dal neurone
class Neuron {
public:
    Neuron();
    const void setInput(const std::vector<double>& input);
    const void setBias(double bias);
    const double getOutput();
    const void setWeight(const std::vector<double>& weight);
    std::vector<double> input_;
    double bias_;
    std::vector<double> weight_;
};

//strato neuronale
class NeuralLayer {
public:
    NeuralLayer(int numNeurons);
    const void setInput(const std::vector<double>& inputs);
    const void setWeight(const std::vector<std::vector<double>>& weights);
    const void setBias(const std::vector<double>& biases);
    const std::vector<double> getOutput();
    std::vector<Neuron> neurons_;

private:
    int numNeurons_;
};

//rete neurale
class NeuralNetwork {
public:
    NeuralNetwork(std::vector<int>& NeuronsPerLayer); 
    const void setInput(const std::vector<double>& input);
    void setWeight(const std::vector<std::vector<double>>& weights);
    void setBias(const std::vector<double>& biases);
    const std::vector<double> getOutput();
    std::vector<NeuralLayer> layers_;
private:
    std::vector<int> NeuronsPerLayer_;

};

#endif