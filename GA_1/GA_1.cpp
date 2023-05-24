#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>


using namespace std;


std::random_device rd;
std::mt19937 gen(rd());

int node = 0;

// Structure to represent an edge

struct Edge {
    int u, v;
    int weight;
};

// Function to calculate the sum of weights for a given set of vertices
int calculateWeightSum(const std::vector<int>& vertices, const std::vector<Edge>& edges) {
    int weightSum = 0;
    for (const auto& edge : edges) {
        if (std::find(vertices.begin(), vertices.end(), edge.u) != vertices.end() &&
            std::find(vertices.begin(), vertices.end(), edge.v) == vertices.end()) {
            weightSum += edge.weight;
        }
    }
    return weightSum;
}

// Genetic Algorithm function
std::vector<int> geneticAlgorithm(const std::vector<Edge>& edges, int numVertices, int populationSize, int maxIterations, int timeLimit) {
    // Create initial population randomly
    std::vector<std::vector<int>> population(populationSize);
    for (int i = 0; i < populationSize; ++i) {
        for (int j = 1; j <= numVertices; ++j) {
            population[i].push_back(j);
        }
        std::shuffle(population[i].begin(), population[i].end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
        population[i].resize(numVertices / 2);

    }
    // Track the best solution found

    std::vector<int> bestSolution;
    int bestWeightSum = 0;



    auto startTime = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count() < timeLimit) {
        // Evaluate fitness of each individual in the population
        std::vector<int> fitness(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            fitness[i] = calculateWeightSum(population[i], edges);
        }

        // Find the best individual in the current  population
        auto maxFitness = std::max_element(fitness.begin(), fitness.end());
        int maxFitnessIndex = std::distance(fitness.begin(), maxFitness);

        // Update the best solution if a better one is found
        if (*maxFitness > bestWeightSum) {
            bestSolution = population[maxFitnessIndex];
            bestWeightSum = *maxFitness;
        }

        // Select parents for reproduction using tournament selection
        std::vector<std::vector<int>> parents(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            std::uniform_real_distribution<int> randomIndex(0, populationSize - 1);
            int index1 = randomIndex(gen);
            int index2 = randomIndex(gen);
            parents[i] = (fitness[index1] > fitness[index2]) ? population[index1] : population[index2];
        }

        // Perform crossover and mutation to create new offspring
        std::vector<std::vector<int>> offspring(populationSize); 
        for (int i = 0; i < populationSize; i += 2) {
            std::uniform_real_distribution<int> randomIndex(0, numVertices / 2 - 1);
            // Perform crossover
            int crossoverPoint = randomIndex(gen);
            offspring[i] = parents[i];
            offspring[i + 1] = parents[i + 1];
            for (int j = crossoverPoint; j < numVertices / 2; ++j) {
                std::swap(offspring[i][j], offspring[i + 1][j]);
            }

            // Perform mutation
            std::uniform_real_distribution<double> randomProbability(0.0, 1.0);
            double mutationProbability = 0.01;  // Adjust the mutation probability as needed
            if (randomProbability(gen) < mutationProbability) {
                int mutationIndex = randomIndex(gen);
                offspring[i][mutationIndex] = (offspring[i][mutationIndex] == mutationIndex + 1) ? mutationIndex + 1 + numVertices / 2 : mutationIndex + 1;
                offspring[i + 1][mutationIndex] = (offspring[i + 1][mutationIndex] == mutationIndex + 1 + numVertices / 2) ? mutationIndex + 1 : mutationIndex + 1 + numVertices / 2;
            }
        }

        // Replace the old population with the new offspring
        population = offspring;
    }

    return bestSolution;
}

int main() {
    std::ifstream inputFile("maxcutin.txt");
    std::ofstream outputFile("maxcutout.txt");

    int numVertices, numEdges;
    inputFile >> numVertices >> numEdges;

    std::vector<Edge> edges(numEdges);
    for (int i = 0; i < numEdges; ++i) {
        inputFile >> edges[i].u >> edges[i].v >> edges[i].weight;
    }

    int populationSize = 100;  // Adjust the population size as needed
    int maxIterations = 1000;  // Adjust the maximum number of iterations as needed
    int timeLimit = 180;       // Adjust the time limit as needed

    std::vector<int> bestSolution = geneticAlgorithm(edges, numVertices, populationSize, maxIterations, timeLimit);

    for (int vertex : bestSolution) {
        outputFile << vertex << " ";
    }
    outputFile << std::endl;

    inputFile.close();
    outputFile.close();

    return 0;
}

