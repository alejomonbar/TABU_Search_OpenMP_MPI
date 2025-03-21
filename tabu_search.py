from mpi4py import MPI
import numpy as np
import random
import time


class ParallelTabuSearch:
    def __init__(self, problem_size, obj_func, tabu_size=10, max_iterations=100):
        """
        Initialize the Parallel Tabu Search algorithm.
        
        Args:
            problem_size: Dimension of the solution
            obj_func: Objective function to minimize
            tabu_size: Size of the tabu list
            max_iterations: Maximum number of iterations
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        print(f"rank: {rank} | size: {size}")
        
        self.problem_size = problem_size
        self.obj_func = obj_func
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.tabu_list = []
        
    def generate_initial_solution(self):
        """Generate a random initial solution."""
        return np.random.randint(0, 2, self.problem_size)
    
    def generate_neighborhood(self, solution, neighborhood_size):
        """Generate neighborhood by flipping bits."""
        neighborhood = []
        for _ in range(neighborhood_size):
            neighbor = solution.copy()
            flip_index = random.randint(0, self.problem_size - 1)
            neighbor[flip_index] = 1 - neighbor[flip_index]  # Flip the bit
            neighborhood.append((neighbor, flip_index))
        return neighborhood
        
    def is_tabu(self, move):
        """Check if a move is in the tabu list."""
        return move in self.tabu_list
    
    def update_tabu_list(self, move):
        """Update the tabu list with the new move."""
        if len(self.tabu_list) >= self.tabu_size:
            self.tabu_list.pop(0)  # Remove oldest tabu move
        self.tabu_list.append(move)
    
    def run(self):
        """Run the parallel tabu search algorithm."""
        # Root process initializes the search
        if self.rank == 0:
            current_solution = self.generate_initial_solution()
            current_value = self.obj_func(current_solution)
            best_solution = current_solution.copy()
            best_value = current_value
            print(f"Initial solution value: {current_value}")
        else:
            current_solution = None
            current_value = None
            best_solution = None
            best_value = None
        
        # Broadcast initial solution to all processes
        current_solution = self.comm.bcast(current_solution, root=0)
        current_value = self.comm.bcast(current_value, root=0)
        best_solution = self.comm.bcast(best_solution, root=0)
        best_value = self.comm.bcast(best_value, root=0)
        
        iteration = 0
        while iteration < self.max_iterations:
            # Each process generates and evaluates a part of the neighborhood
            neighborhood_size = 10 * self.size  # Total neighborhood size
            local_size = neighborhood_size // self.size  # Local part
            
            # Generate local part of neighborhood
            local_neighborhood = self.generate_neighborhood(
                current_solution, 
                local_size + (1 if self.rank < neighborhood_size % self.size else 0)
            )
            
            # Evaluate neighbors
            local_best_neighbor = None
            local_best_value = float('inf')
            local_best_move = None
            
            for neighbor, move_index in local_neighborhood:
                if not self.is_tabu(move_index):
                    value = self.obj_func(neighbor)
                    if value < local_best_value:
                        local_best_neighbor = neighbor
                        local_best_value = value
                        local_best_move = move_index
            # Prepare local results for reduction
            local_result = [local_best_value, local_best_neighbor, local_best_move]
            if local_best_neighbor is None:
                local_result = [float('inf'), None, None]
            
            # Gather all local results to root
            all_results = self.comm.gather(local_result, root=0)
            
            # Root process selects the best neighbor
            if self.rank == 0:
                global_best_value = float('inf')
                global_best_neighbor = None
                global_best_move = None
                
                for result in all_results:
                    value, neighbor, move = result
                    if neighbor is not None and value < global_best_value:
                        global_best_value = value
                        global_best_neighbor = neighbor
                        global_best_move = move
                
                # If no non-tabu move improves the solution, choose the best tabu move (aspiration)
                if global_best_neighbor is None:
                    # Generate some random neighbors without tabu restrictions
                    aspiration_neighbors = self.generate_neighborhood(current_solution, 5)
                    for neighbor, move_index in aspiration_neighbors:
                        value = self.obj_func(neighbor)
                        if value < global_best_value:
                            global_best_value = value
                            global_best_neighbor = neighbor
                            global_best_move = move_index
                
                # Update current solution
                if global_best_neighbor is not None:
                    current_solution = global_best_neighbor
                    current_value = global_best_value
                    self.update_tabu_list(global_best_move)
                    
                    # Update best solution if improved
                    if current_value < best_value:
                        best_solution = current_solution.copy()
                        best_value = current_value
                        print(f"Iteration {iteration}: New best value = {best_value} Rank: {self.rank}")
            
            # Broadcast updated information to all processes
            current_solution = self.comm.bcast(current_solution, root=0)
            current_value = self.comm.bcast(current_value, root=0)
            
            # Synchronize tabu list across all processes
            self.tabu_list = self.comm.bcast(self.tabu_list if self.rank == 0 else None, root=0)
            
            iteration += 1
        
        # Gather final results
        if self.rank == 0:
            print(f"Search completed after {self.max_iterations} iterations")
            print(f"Best solution found: {best_solution}")
            print(f"Best value: {best_value}")
            return best_solution, best_value
        return None, None


# Example usage: Solve a binary optimization problem
def example_objective_function(solution):
    """
    Example objective function (minimize):
    Try to get all 1s in the solution, with a penalty for consecutive 1s
    """
    # Basic sum of elements (wanting all 1s)
    base_value = -np.sum(solution)
    
    # Penalty for consecutive 1s
    penalty = 0
    for i in range(len(solution) - 1):
        if solution[i] == 1 and solution[i + 1] == 1:
            penalty += 2
    
    return base_value + penalty


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Problem parameters
    problem_size = 50
    tabu_size = 15
    max_iterations = 100
    
    start_time = time.time()
    
    # Create and run the parallel tabu search
    tabu_search = ParallelTabuSearch(
        problem_size=problem_size,
        obj_func=example_objective_function,
        tabu_size=tabu_size,
        max_iterations=max_iterations
    )
    
    solution, value = tabu_search.run()
    
    if rank == 0:
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")