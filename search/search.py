# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    # Start: A
    # Is the start a goal? False
    # Start's successors: [('B', '0:A->B', 1.0), ('C', '1:A->C', 2.0), ('D', '2:A->D', 4.0)]
                         # (successor, action, stepCost)

    # Return: ['2:A->B2', '0:B2->C', '0:C->D', '2:D->E2', '0:E2->F', '0:F->G']
    # piazza: add paths to stack -- add tuple (node, path to that node)

    visited_nodes = []
    path = []
    stack = util.Stack()

    stack.push((problem.getStartState(), path)) # Add start state and path to stack
    
    while not stack.isEmpty():
        # Pop node from stack
        node_and_path = stack.pop()
        
        curr_node = node_and_path[0]        
        curr_path = node_and_path[1]

        # check if current node is the goal
        if problem.isGoalState(curr_node):
            return curr_path

        # check if curent node has been visited
        if (curr_node not in visited_nodes):
            visited_nodes.append(curr_node)
                        
            #   loop through adjacent nodes, 
            #       if visited -> ignore
            #       not visited -> add to stack

            for triple in problem.getSuccessors(curr_node):
                if (triple[0] not in visited_nodes):
                    successor_path = curr_path.copy()
                    successor_path.append(triple[1])
                    stack.push((triple[0], successor_path))

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    visited_nodes = []
    path = []
    queue = util.Queue()

    queue.push((problem.getStartState(), path)) # Add start state and path to queue

    while not queue.isEmpty():
        # Pop node from queue
        node_and_path = queue.pop()
        
        curr_node = node_and_path[0]        
        curr_path = node_and_path[1]

        # check if current node is the goal
        if problem.isGoalState(curr_node):
            return curr_path

        # check if curent node has been visited
        if (curr_node not in visited_nodes):
            visited_nodes.append(curr_node)
                        
            #   loop through adjacent nodes, 
            #       if visited -> ignore
            #       not visited -> add to queue

            for triple in problem.getSuccessors(curr_node):
                if (triple[0] not in visited_nodes):
                    successor_path = curr_path.copy()
                    successor_path.append(triple[1])
                    queue.push((triple[0], successor_path))
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # From start state -> visit adjacent nodes -> choose node w/ least cost
    # From all the unvisited nodes, adjacent nodes -> choose node w/ least cost
    # Iterate through all possible paths even if we reach the goal state

    # Return the actions path which gives the least total cost

    visited_nodes = []
    path = []
    priority_queue = util.PriorityQueue()  # Create the priority queue
    priority_queue.push((problem.getStartState(), path), 0) # Add the start state and cost ((state, path), cost)

    while not priority_queue.isEmpty():
        node_and_path = priority_queue.pop()

        curr_node = node_and_path[0]
        curr_path = node_and_path[1]
        curr_cost = problem.getCostOfActions(curr_path)

        # check if current node is the goal
        if problem.isGoalState(curr_node):
            return curr_path
                
        if curr_node not in visited_nodes:
            visited_nodes.append(curr_node)
            
            for triple in problem.getSuccessors(curr_node):
                if (triple[0] not in visited_nodes):
                    # priority_queue.update(triple[0], triple[2]) # update pq if node is already in the pq
                    cumulative_cost = curr_cost + triple[2]
                    successor_path = curr_path.copy()
                    successor_path.append(triple[1])
                    priority_queue.push((triple[0], successor_path), cumulative_cost)
                    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # f(n) = g(n) + h(n); the estimated cost from start -> goal
    # g(n) - total cost from start -> current node (backward cost)
    # h(n) - total cost from current node -> goal (forward cost)

    # A* terminates when we dequeue a goal (ensures lowest combined cost and heuristic)

    visited_nodes = [] # Nodes that have been visited
    path = []
    priority_queue = util.PriorityQueue()  # Create the priority queue
    priority_queue.push((problem.getStartState(), path), 0) # Push the start state and cost ((state, path), cost)
    
    while not priority_queue.isEmpty():
        node_and_path = priority_queue.pop()

        curr_node = node_and_path[0]
        curr_path = node_and_path[1]
        curr_cost = problem.getCostOfActions(curr_path)

        # check if current node is the goal
        if problem.isGoalState(curr_node):
            return curr_path
                
        if curr_node not in visited_nodes:
            visited_nodes.append(curr_node)
            
            for triple in problem.getSuccessors(curr_node):
                if (triple[0] not in visited_nodes):
                    cumulative_cost = curr_cost + triple[2]
                    successor_path = curr_path.copy()
                    successor_path.append(triple[1])
                    priority_queue.push((triple[0], successor_path), cumulative_cost + heuristic(triple[0], problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
