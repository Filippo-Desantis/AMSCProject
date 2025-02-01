#ifndef IMPLEMENTAZIONECLASSI_HPP
#define IMPLEMENTAZIONECLASSI_HPP

#include "funzioni.hpp"

double sigma(double a){
    return 1 / (1 + std::exp(-a));
}
// Implementazione di Neuron, impostazione di input, pesi, bias e output
Neuron::Neuron() : input_(0.0), weight_(1.0), bias_(0.0) {}

const void Neuron::setInput(const std::vector<double>& input) { 
    input_ = input;
}

const double Neuron::getOutput() {
    //return std::tanh(dotproduct(weight_,input_) - bias_);
    return sigma(dotproduct(weight_,input_) - bias_);
}

const void Neuron::setWeight(const std::vector<double>& weight) {
    weight_ = weight;
}

const void Neuron::setBias(double bias) {
    bias_ = bias;
}

// Implementazione di NeuralLayer
NeuralLayer::NeuralLayer(int numNeurons)
    : numNeurons_(numNeurons) {
    for (int i = 0; i < numNeurons_; ++i) {
        neurons_.emplace_back();
    }
}

const void NeuralLayer::setInput(const std::vector<double>& inputs) {
    //#pragma omp parallel for
    for (int i = 0; i < numNeurons_; ++i) {
        neurons_[i].setInput(inputs);
    }
}

const void NeuralLayer::setWeight(const std::vector<std::vector<double>>& weights) {
    for (int i = 0; i < numNeurons_; ++i) {
        neurons_[i].setWeight(weights[i]);
    }
}

const void NeuralLayer::setBias(const std::vector<double>& biases) {
    for (int i = 0; i < numNeurons_; ++i) {
        neurons_[i].setBias(biases[i]);
    }
}


const std::vector<double> NeuralLayer::getOutput() {
    std::vector<double> outputs(numNeurons_);
    //#pragma omp parallel for
    for (int i = 0; i < numNeurons_; ++i) {
        outputs[i] = neurons_[i].getOutput();
    }
    return outputs;
}

// Implementazione di NeuralNetwork
NeuralNetwork::NeuralNetwork(std::vector<int>& NeuronsPerLayer) : NeuronsPerLayer_(NeuronsPerLayer) {
    for (int i = 0; i < NeuronsPerLayer_.size(); ++i) {
        layers_.emplace_back(NeuronsPerLayer_[i]);
    }
}

const void NeuralNetwork::setInput(const std::vector<double>& input) {
    layers_[0].setInput(input);
}

void NeuralNetwork::setWeight(const std::vector<std::vector<double>>& weights){
    std::vector<std::vector<double>> w; 
    unsigned int acc = 0;
    for(unsigned int i = 0; i < NeuronsPerLayer_.size(); ++i){
        for(unsigned int j = 0; j < NeuronsPerLayer_[i]; ++j){
            w.emplace_back(weights[j + acc]);
        }
        layers_[i].setWeight(w);
        acc += NeuronsPerLayer_[i];
        w.clear();
    }
}

void NeuralNetwork::setBias(const std::vector<double>& bias){
    std::vector<double> b; 
    unsigned int acc = 0;
    for(unsigned int i = 0; i < NeuronsPerLayer_.size(); ++i){
        for(unsigned int j = 0; j < NeuronsPerLayer_[i]; ++j){
            b.emplace_back(bias[j + acc]);
        }
        layers_[i].setBias(b);
        acc += NeuronsPerLayer_[i];
        b.clear();
    }
}

const std::vector<double> NeuralNetwork::getOutput() {
    for(unsigned int i = 1; i < NeuronsPerLayer_.size(); ++i){
        layers_[i].setInput(layers_[i-1].getOutput());
    }
    //return layers_[NeuronsPerLayer_.size()-1].getOutput();
    return {(layers_[NeuronsPerLayer_.size()-2].getOutput().back() + 1.0)*0.5};
}


#endif