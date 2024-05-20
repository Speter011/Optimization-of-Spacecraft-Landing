package coursework;

import java.util.ArrayList;
import java.util.Random;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population. Currently returns random Individual
			Individual parent1 = select(); 
			Individual parent2 = select();

			// Generate a child by crossover. Not Implemented			
			ArrayList<Individual> children = reproduce(parent1, parent2);			
			
			//mutate the offspring
			mutate(children);
			
			// Evaluate the children
			evaluateIndividuals(children);			

			// Replace children in population
			replace(children);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * member of the population
	 * 
	 *Best solution: Roulette wheel
	 * 
	 */
	private Individual select() {
		

		
//		//tournament selection operation
//		int tournamentSize = 15; 
//		Individual[] potentialParent = new Individual[tournamentSize];
		Individual chosenParent = new Individual();
//		
//		Double bestFitness;
//		for (int i = 0; i < tournamentSize ; i++) {
//			potentialParent[i] = population.get(Parameters.random.nextInt(Parameters.popSize));
//			
//		}
//		
//		bestFitness = potentialParent[0].fitness;	
//		for (int i = 1; i < tournamentSize; i++) {
//			if (potentialParent[i].fitness > bestFitness)
//					{
//						bestFitness = potentialParent[i].fitness ;
//						chosenParent = potentialParent[i];
//					}
//		}
	
		//Roulette wheel selection
		double probability;
		double fitnessSum = 0;
		double allProbs;
		int id;
		Random rand = new Random();
		
		for(int i = 0; i < Parameters.popSize; i++){
			fitnessSum += population.get(i).fitness;
		}
		
		
		for(int i = 0; i < Parameters.popSize; i++){			
			if(rand.nextDouble() < (population.get(i).fitness / fitnessSum)){
				chosenParent = population.get(i);
				id = i;
			}
		}
		
		//random selection
		//chosenParent = population.get(Parameters.random.nextInt(Parameters.popSize));
		
		
		return chosenParent.copy();
		//return id;
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * NEEDS REPLACED with proper method this code just returns exact copies of the
	 * parents. 
	 */
	
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		
		
		
    	ArrayList<Individual> children = new ArrayList<>();
    	Random rand = new Random();
    	int cutPoint;
    	int cutPoint2;
//		children.add(parent1.copy());
//		children.add(parent2.copy());
    	
    	// pick cut point
    	int chromosomeLength = parent1.chromosome.length;
    	cutPoint = rand.nextInt(chromosomeLength);
      	//create empty child array
    	
    	Individual child = new Individual();
    	Individual child2 = new Individual();
    	
    	
    	// genes from parent1
    	for (int i = 0; i < cutPoint; i++) {
    		child.chromosome[i] = parent1.chromosome[i];
    		child2.chromosome[i] = parent2.chromosome[i];
    	}
    	 
    	
    	// and genes from parent2
    	for (int i = cutPoint; i < chromosomeLength; i++) {
    		child2.chromosome[i] = parent1.chromosome[i];
    		child.chromosome[i] = parent2.chromosome[i];
    	}
    	
    	
    	/*
    	for(int i = 0; i< chromosomeLength; i++) {
    		if(rand.nextBoolean()) {
    			child.chromosome[i] = parent1.chromosome[i];
    			child2.chromosome[i] = parent2.chromosome[i];
    		}
    		else {
    			child.chromosome[i] = parent2.chromosome[i];
    			child2.chromosome[i] = parent1.chromosome[i];
    		}
    	}
    	*/ 	
       	children.add(child);
    	children.add(child2);
    	return children;
		
	} 
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}

	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
//		
		//replace worst
		for(Individual individual : individuals) {
			int idx = getWorstIndex();		
			population.set(idx, individual);
		}
//
		
//		//replace random
//		for(Individual individual : individuals) {
//			int idx = Parameters.random.nextInt(Parameters.popSize);
//			population.set(idx, individual);
//		}
		
}
		
		//tournament replacement operation
//		int tournamentSize = 3;
//		Individual[] potentialPeople = new Individual[tournamentSize]; 
//		Individual[] chosenReplace = new Individual[2];
//		Double worstFitness = null;
//		for (int i = 0; i < tournamentSize ; i++) {
//			potentialPeople[i] = population.get(Parameters.random.nextInt(Parameters.popSize));
//		}
//			
//		for (int i = 1; i < tournamentSize; i++) {
//			if (potentialPeople[i].fitness > worstFitness)
//					{
//						worstFitness = potentialPeople[i].fitness ;
//						chosenReplace[0] = potentialPeople[i];
//						chosenReplace[1] = potentialPeople[i];
//					}
//		}			

	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		
		//sigmoid function
		//return (1 / (1 + Math.exp(-x)) - 0.5);
		
		//ReLU function
		return (Math.max(0, x) - 0.5);
		
		//tanh function
		//return Math.tanh(x);
		
		//linear
		//return x;
		
	}
}
