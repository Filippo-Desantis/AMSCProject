//#include "versione2.hpp"
#include "training.hpp"
int main() {
    auto start = std::chrono::high_resolution_clock::now();

    //costruzione della rete neurale
    std::vector<int> neuronsperlayer = {2,4,3,1}; 
    NeuralNetwork neuralNetwork(neuronsperlayer);

    
    //Generazione casuale dei pesi 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    std::vector<std::vector<double>> weights;
    for (int k = 0; k < neuronsperlayer.size(); ++k) {
        std::vector<std::vector<double>> layer_weights;
        for (int i = 0; i < neuronsperlayer[k]; ++i) {
            std::vector<double> neuron_weights;
            if (k != 0) {
                for (int j = 0; j < neuronsperlayer[k - 1]; ++j) {
                    neuron_weights.push_back(distribution(gen));
                }
            } else {
                for (int j = 0; j < 2; ++j) { // Numero di input = 2, l'ho inserito "manualmente" poiché farò cambiare gli input nei cicli annidati successivi
                    neuron_weights.push_back(distribution(gen));
                }
            }
            layer_weights.push_back(neuron_weights);
        }
        weights.insert(weights.end(), layer_weights.begin(), layer_weights.end());
    }

    //std::vector<std::vector<double>> weights = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    neuralNetwork.setWeight(weights);

    //inizializzo i bias a zero
    std::vector<double> biases = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    neuralNetwork.setBias(biases);

    //visualizziamo i risultati pre-training
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {

            std::vector<double> input = {static_cast<double>(a), static_cast<double>(b)};
            neuralNetwork.setInput(input);

            //ottengo e stampo l'output (uno scalare, o meglio, un vettore con un unico elemento nel caso dell'output layer)
            std::vector<double> output = neuralNetwork.getOutput();

            std::cout << a << " XOR " << b << " --> " << output.back() << std::endl;
        }
    }


    // scrivo i training data per la funzione di training (input1, input2, target)
    //uso dei double in quanto inizialmente avevo intenzione di fare una rete neurale più generale possibile.
    std::vector<std::vector<double>> trainingData = {
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0}
    };

    unsigned int epochs = 2000; // numero di epoche
    trainNeuralNetwork(neuralNetwork, trainingData, epochs);

   std::cout<<std::endl;

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {

            std::vector<double> input = {static_cast<double>(a), static_cast<double>(b)};
            neuralNetwork.setInput(input);
            std::vector<double> output = neuralNetwork.getOutput();
            std::cout << a << " XOR " << b << " --> ";
            printVector(output);
            //std::cout << a << " XOR " << b << " --> " << output.back() << std::endl;

        }
    }
    //visualizziamo il tempo di esecuzione
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Tempo di esecuzione: " << duration.count() << " ms" << std::endl;

    return 0;
}
