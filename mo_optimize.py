import numpy as np
import os
import pickle
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.core.evaluator import Evaluator
from pymoo.config import Config
Config.warnings['not_compiled'] = False

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
        
    def setup_algorithm(self):
        """Set up the NSGA-II algorithm"""
        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=self.initial_population if self.initial_population is not None else FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20, prob=self.mutation_num_genes/self.n_var),
            eliminate_duplicates=True
        )
        
    def custom_evaluation(self, problem, pop, algorithm):
        """
        Custom evaluation function that directly calls the fitness_func for each solution.
        This completely bypasses pymoo's standard evaluation process.
        """
        print("\n===== CUSTOM EVALUATION STARTING =====")
        
        # Get the solutions to be evaluated
        X = pop.get("X")
        n_solutions = len(X)
        
        # Create array to store results
        F = np.zeros((n_solutions, 2))
        
        for i in range(n_solutions):
            print(f"\n----- Evaluating Solution {i} -----")
            print(f"Solution: {X[i]}")
            
            # Directly call the fitness function
            try:
                # Call the user-provided fitness function (passing None as ga_instance)
                start_time = time.time()
                fitness_values = self.fitness_func(None, X[i], i)
                end_time = time.time()
                
                print(f"Solution {i} evaluated in {end_time - start_time:.2f} seconds")
                print(f"Fitness values: {fitness_values}")
                
                # Our fitness maximizes, but pymoo minimizes, so negate
                F[i, 0] = -fitness_values[0]
                F[i, 1] = -fitness_values[1]
            except Exception as e:
                print(f"Error evaluating solution {i}: {e}")
                import traceback
                traceback.print_exc()
                F[i, 0] = 1000  # High value for minimization
                F[i, 1] = 1000
        
        # Set the fitness values in the population
        pop.set("F", F)
        
        print("\n===== CUSTOM EVALUATION COMPLETE =====")
        return pop
        
    def run(self):
        """Run the optimization algorithm."""
        print("\n===== STARTING OPTIMIZATION RUN =====")
        
        # Create a fresh algorithm
        self.setup_algorithm()
        
        # Create problem definition
        problem = Problem(
            n_var=self.n_var,
            n_obj=2,
            n_constr=0,
            xl=self.xl,
            xu=self.xu
        )
        
        # Run just one generation
        termination = MaximumGenerationTermination(1)
        
        # Override the standard evaluator with our custom one
        evaluator = Evaluator()
        evaluator.eval = lambda problem, pop, **kwargs: self.custom_evaluation(problem, pop, self.algorithm)
        
        # Run the algorithm
        print("Starting NSGA-II run...")
        res = minimize(
            problem,
            self.algorithm,
            termination,
            seed=1,
            verbose=True,
            save_history=True,
            evaluator=evaluator
        )
        
        print("Optimization completed!")
        
        # Store results
        self.current_generation += 1
        self.result = res
        
        if res.X is not None and len(res.X) > 0:
            print("Storing results...")
            # Store the final population
            self.last_generation_parents = res.pop.get("X")
            self.last_generation_fitness = -res.pop.get("F")  # Convert back to maximization
            
            # Extract Pareto fronts
            try:
                ranks = res.pop.get("rank")
                self.pareto_fronts = []
                for i in range(int(np.max(ranks)) + 1):
                    front = np.where(ranks == i)[0]
                    self.pareto_fronts.append(front.tolist())
            except:
                # Fallback if rank information not available
                print("Using simple Pareto front extraction")
                self.pareto_fronts = [[i for i in range(len(self.last_generation_parents))]]
            
            # Set initial population for next run
            self.initial_population = self.last_generation_parents
        else:
            print("Warning: No valid results from optimization!")
            
        # Call generation callback
        if self.on_generation:
            self.on_generation(self)
            
        return res
    
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
            
        if fitness is None or len(fitness) == 0:
            print("Warning: No fitness values available")
            return None, None, None
        
        # For multi-objective, we choose one solution from the Pareto front
        # Here we prioritize the first objective (trajectory fitness)
        idx = np.argmax(fitness[:, 0])
        solution = self.last_generation_parents[idx]
        solution_fitness = fitness[idx]
        
        return solution, solution_fitness, idx
    
    def save(self, filename):
        """
        Save the optimizer state to a file.
        
        Args:
            filename: Name of the file to save to (without extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        data = {
            "last_generation_parents": self.last_generation_parents.tolist() if self.last_generation_parents is not None else None,
            "last_generation_fitness": self.last_generation_fitness.tolist() if self.last_generation_fitness is not None else None,
            "pareto_fronts": self.pareto_fronts,
            "n_var": self.n_var,
            "xl": self.xl.tolist(),
            "xu": self.xu.tolist(),
            "current_generation": self.current_generation,
            "pop_size": self.pop_size,
        }
        
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
                num_generations=1,
                pop_size=data.get("pop_size", 40)
            )
            
            # Set loaded state
            optimizer.n_var = data["n_var"]
            optimizer.xl = np.array(data["xl"])
            optimizer.xu = np.array(data["xu"])
            optimizer.last_generation_parents = np.array(data["last_generation_parents"]) if data["last_generation_parents"] is not None else None
            optimizer.last_generation_fitness = np.array(data["last_generation_fitness"]) if data["last_generation_fitness"] is not None else None
            optimizer.pareto_fronts = data["pareto_fronts"]
            optimizer.current_generation = data["current_generation"]
            
            # Set initial population for next run
            optimizer.initial_population = optimizer.last_generation_parents
            
            print(f"Optimization state loaded from {filename}.pkl")
            return optimizer
        except FileNotFoundError:
            raise FileNotFoundError(f"Error reading the file {filename}. Please check your inputs.")
        except Exception as e:
            raise Exception(f"Error loading optimization state: {e}")