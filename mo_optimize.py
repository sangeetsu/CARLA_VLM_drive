import numpy as np
import os
import pickle
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer using NSGA-II algorithm from pymoo.
    Designed as a drop-in replacement for PyGAD GA class.
    """
    
    def __init__(
        self,
        num_generations=100,
        pop_size=40,
        num_parents_mating=None,
        initial_population=None,
        fitness_func=None,
        on_generation=None,
        gene_space=None,
        mutation_num_genes=2,
        parent_selection_type="tournament",
    ):
        """Initialize the multi-objective optimizer."""
        self.num_generations = num_generations
        self.pop_size = pop_size if initial_population is None else len(initial_population)
        self.num_parents_mating = num_parents_mating if num_parents_mating is not None else self.pop_size // 2
        self.fitness_func = fitness_func
        self.on_generation = on_generation
        self.initial_population = initial_population
        self.mutation_num_genes = mutation_num_genes
        
        # Convert gene_space to proper bounds
        if gene_space is not None:
            self.xl = np.array([g['low'] for g in gene_space])
            self.xu = np.array([g['high'] for g in gene_space])
            self.n_var = len(gene_space)
        else:
            self.xl = np.array([0, 0, 0, 0, 0, 0, 0, -15])  # Default lower bounds
            self.xu = np.array([5, 5, 5, 5, 5, 5, 5, 15])   # Default upper bounds
            self.n_var = 8  # Default number of variables
        
        # Store results and state
        self.result = None
        self.current_generation = 0
        self.last_generation_fitness = None
        self.last_generation_parents = None
        self.pareto_fronts = None
        
        # Initialize algorithm
        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=FloatRandomSampling() if self.initial_population is None else None,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20, prob=mutation_num_genes/self.n_var),
            selection=TournamentSelection(func_comp=self.tournament_selection),
            eliminate_duplicates=True
        )
        
        # Create problem definition
        self.problem = None  # Will be set in run()

    def tournament_selection(self, pop, P, **kwargs):
        """Custom tournament selection that respects parent_selection_type."""
        return P
    
    def _evaluate_fitness(self, x, out):
        """
        Evaluate the fitness function for each solution.
        This is called by pymoo's optimization process.
        """
        n_solutions = x.shape[0]
        F = np.zeros((n_solutions, 2))  # Assuming 2 objectives: trajectory and velocity
        
        for i in range(n_solutions):
            solution = x[i, :]
            # Call the original fitness function
            fit_values = self.fitness_func(self, solution, i)
            # Our fitness maximizes, but pymoo minimizes, so negate
            F[i, 0] = -fit_values[0]  # Trajectory fitness
            F[i, 1] = -fit_values[1]  # Velocity fitness
        
        out["F"] = F
    
    def callback(self, algorithm):
        """Callback after each generation."""
        self.current_generation += 1
        
        # Store current population and fitness
        self.last_generation_parents = algorithm.pop.get("X")
        self.last_generation_fitness = -algorithm.pop.get("F")  # Negate to convert back to maximization
        
        # Extract Pareto fronts
        ranks = algorithm.pop.get("rank")
        self.pareto_fronts = []
        for front_idx in range(int(np.max(ranks))+1):
            front = np.where(ranks == front_idx)[0]
            self.pareto_fronts.append(front.tolist())
        
        # Call the user's callback if provided
        if self.on_generation:
            self.on_generation(self)
        
        print(f"Generation {self.current_generation}/{self.num_generations} completed")
    
    def run(self):
        """Run the optimization algorithm."""
        # Create problem definition
        self.problem = Problem(
            n_var=self.n_var,
            n_obj=2,  # Two objectives: trajectory and velocity
            n_constr=0,  # No constraints
            xl=self.xl,
            xu=self.xu,
            evaluation_of=self._evaluate_fitness
        )
        
        # If we have an initial population, set it
        if self.initial_population is not None:
            # Ensure initial population respects bounds
            self.initial_population = np.clip(self.initial_population, self.xl, self.xu)
            self.algorithm.initialization.sampling = self.initial_population
        
        # Run optimization
        termination = MaximumGenerationTermination(self.num_generations)
        self.result = minimize(
            self.problem,
            self.algorithm,
            termination,
            callback=self.callback,
            verbose=True,
            save_history=True
        )
        
        # Store final results
        self.last_generation_parents = self.result.X
        self.last_generation_fitness = -self.result.F  # Negate to convert back to maximization
        
        # Extract final Pareto front
        ranks = self.algorithm.pop.get("rank")
        self.pareto_fronts = []
        for front_idx in range(int(np.max(ranks))+1):
            front = np.where(ranks == front_idx)[0]
            self.pareto_fronts.append(front.tolist())
        
        print("Optimization completed!")
        return self.result
    
    def best_solution(self, last_generation_fitness=None):
        """
        Get the best solution from the last generation.
        
        Returns:
            solution: The best solution
            fitness: Fitness of the best solution
            idx: Index of the best solution
        """
        if last_generation_fitness is None:
            fitness = self.last_generation_fitness
        else:
            fitness = last_generation_fitness
        
        # For multi-objective, we choose one solution from the Pareto front
        # Here we prioritize the first objective (trajectory fitness)
        idx = np.argmax(fitness[:, 0])
        solution = self.last_generation_parents[idx]
        solution_fitness = fitness[idx]
        
        return solution, solution_fitness, idx
    
    def save(self, filename):
        """
        Save the optimizer state to a file without freezing.
        
        Args:
            filename: Name of the file to save to (without extension)
        """
        # Create a dictionary with minimal necessary data
        data = {
            "last_generation_parents": self.last_generation_parents.tolist() if self.last_generation_parents is not None else None,
            "last_generation_fitness": self.last_generation_fitness.tolist() if self.last_generation_fitness is not None else None,
            "pareto_fronts": self.pareto_fronts,
            "n_var": self.n_var,
            "xl": self.xl.tolist(),
            "xu": self.xu.tolist(),
            "current_generation": self.current_generation,
        }
        
        # Save using pickle
        try:
            print(f"Saving state to {filename}.pkl...")
            with open(f"{filename}.pkl", "wb") as f:
                pickle.dump(data, f)
            print(f"Optimization state saved to {filename}.pkl")
            return True
        except Exception as e:
            print(f"Error saving optimization state: {e}")
            return False
    
    @classmethod
    def load(cls, filename):
        """
        Load optimizer state from a file.
        
        Args:
            filename: Name of the file to load from (without extension)
            
        Returns:
            A MultiObjectiveOptimizer instance with loaded state
        """
        try:
            with open(f"{filename}.pkl", "rb") as f:
                data = pickle.load(f)
            
            # Create a new instance
            optimizer = cls(
                num_generations=100,  # Default values, will be overwritten
                pop_size=40
            )
            
            # Set loaded state
            optimizer.n_var = data["n_var"]
            optimizer.xl = np.array(data["xl"])
            optimizer.xu = np.array(data["xu"])
            optimizer.last_generation_parents = np.array(data["last_generation_parents"]) if data["last_generation_parents"] is not None else None
            optimizer.last_generation_fitness = np.array(data["last_generation_fitness"]) if data["last_generation_fitness"] is not None else None
            optimizer.pareto_fronts = data["pareto_fronts"]
            optimizer.current_generation = data["current_generation"]
            
            print(f"Optimization state loaded from {filename}.pkl")
            return optimizer
        except FileNotFoundError:
            raise FileNotFoundError(f"Error reading the file {filename}. Please check your inputs.")
        except Exception as e:
            raise Exception(f"Error loading optimization state: {e}")