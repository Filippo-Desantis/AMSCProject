#ifndef FUNZIONI_HPP
#define FUNZIONI_HPP

#include "defclassi.hpp"

//qui definiamo alcune funzioni utili successivamente

//implementazione del prodotto scalare fra due vettori
double dotproduct(std::vector<double> a, std::vector<double> b){
    double risultato = 0.0; 
    if(a.size() != b.size()){
    std::cout << "errore nel prodotto fra i vettori, diverse dimensioni" << std::endl;
    }
    else{   
        for(unsigned int i = 0; i<a.size(); ++i)
        risultato += a[i]*b[i];
    }
    return risultato;
}

//implementazione della funzione di perdita "loss quadratic"
double loss_quadratic(double out, double y){
    double error = y - out;
    return error*error*0.5;
}

double grad_quadratic(double out, double y){
    return out - y;
}

//questa funzione mi è servita per controllare alcuni vettori nel debugging
void printVector(const std::vector<double>& vec) {
    std::cout << "[ ";
    for (const double& element : vec) {
        std::cout << element << " ";
    }
    std::cout << "]" << std::endl;
}

#endif