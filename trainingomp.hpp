#ifndef TRAININGOMP_HPP
#define TRAININGOMP_HPP

#include "implementazioneclassi.hpp"


//qui implemento la funzione di training. Ho pensato di fare un gradient descent

//il problema credo sia proprio in questa funzione, infatti nonostante la rete neurale non abbia problemi in sé
//essa, dopo il training, non si avvicina minimamente al risultato corretto della tabella dello XOR, ma nonostante il tempo dedicato al debugging non
//sono riuscito a trovare il baco.

void trainNeuralNetwork(NeuralNetwork& neuralNetwork, const std::vector<std::vector<double>>& trainingData, unsigned int epochs) {
    double learningRate = 1;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (const auto& data : trainingData) {
            double input1 = data[0];
            double input2 = data[1];
            double target = data[2];

            neuralNetwork.setInput({input1, input2});
            std::vector<double> output = neuralNetwork.getOutput();
            // Calcolo della funzione di perdita
            double loss = loss_quadratic(output.back(), target);
            totalLoss += loss;
            std::vector<double> temp;

            // calcolo del gradiente e aggiornamento di pesi e bias
            for (int layerIdx = neuralNetwork.layers_.size() - 1; layerIdx >= 0; --layerIdx) {

                NeuralLayer& layer = neuralNetwork.layers_[layerIdx];
                std::vector<Neuron>& neurons = layer.neurons_;
                std::vector<double> error(neurons.size());

                #pragma omp parallel for 
                for (unsigned int neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx) {

                    //calcolo della derivata della activation function (tanh) per calcolo dell'errore
                    Neuron& neuron = neurons[neuronIdx];
                    //double derivative = 1.0 / (std::cosh(dotproduct(neuron.weight_ , neuron.input_) - neuron.bias_) * std::cosh(dotproduct(neuron.weight_ , neuron.input_) - neuron.bias_));
                    double derivative = sigma(dotproduct(neuron.weight_ , neuron.input_) - neuron.bias_) * (1 - sigma(dotproduct(neuron.weight_ , neuron.input_) - neuron.bias_));
                    if(layerIdx == neuralNetwork.layers_.size()-1) 
                        error[neuronIdx]= (neuron.getOutput() - target)*derivative;

                    else{
                        double sum = 0.0;
                        for(unsigned int i = 0; i < temp.size(); ++i){
                            sum += temp[i] * neuralNetwork.layers_[layerIdx+1].neurons_[i].weight_[neuronIdx];
                        }
                        error[neuronIdx] = sum*derivative;
                        }

                    std::vector<double> input = neuron.input_;
                    std::vector<double> weightUpdates(input.size());

                    #pragma omp parallel for
                    for (unsigned int i = 0; i < input.size(); ++i) {
                        weightUpdates[i] = neuron.weight_[i] - learningRate * error[neuronIdx] * input[i];
                    }

                    neuron.setWeight(weightUpdates);
                    double biasUpdates = neuron.bias_ - learningRate * error[neuronIdx];
                    neuron.setBias(biasUpdates);

                }

                temp = error;
            }
        }

        // per comodità avevo aggiunto questa riga per vedere gli output man mano e la perdita, commentata per snellire la schermata del terminale
        //std::cout << "Total Loss: " << totalLoss << std::endl;
    }
}

#endif