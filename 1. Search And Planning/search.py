# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""
from collections import Set
from operator import contains
from sets import Set

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


class Fringe:
    def __init__(self, state, actions, cost_backward, cost_forward):
        """
        :type state: tuple
        :type actions: list
        :type cost_backward: int
        :type cost_forward: int
        :rtype : Fringe
        """
        self.state = state
        self.actions = actions
        self.cost_backward = cost_backward
        self.cost_forward = cost_forward

    def total_cost(self):
        """
        :return: int
        """
        return self.cost_backward + self.cost_forward


def priority_func_max_cost_first(fringe):
    """
    :returns priority inversely proportional to the given cost
    :param fringe: Fringe
    :rtype : int
    """
    return -fringe.total_cost()


def priority_func_min_cost_first(fringe):
    """
    :returns priority directly proportional to the given cost
    :param fringe: Fringe
    :rtype : int
    """
    return fringe.total_cost()


def graph_search(problem, priority_func, heuristic=None):

    # start fringe (state, actions, cost)
    """
    :param problem: problem
    :param priority_func: func(Fringe) returns int
    :param heuristic: func(state, problem)
    :return: actions for the problem
    """
    goal_fringe = fringe = Fringe(problem.getStartState(), [], 0, 0)

    # closed set
    closed = Set()

    # all possible states
    from util import PriorityQueueWithFunction
    fringe_queue = PriorityQueueWithFunction(priority_func)
    fringe_queue.push(fringe)

    while not fringe_queue.isEmpty():
        # pop
        fringe = fringe_queue.pop()

        # goal test
        if problem.isGoalState(fringe.state):
            goal_fringe = fringe
            break

        # closed set
        if contains(closed, fringe.state):
            continue
        closed.add(fringe.state)

        # expand
        for successor in problem.getSuccessors(fringe.state):
            next_state, action, cost = successor
            expanded_fringe = Fringe(next_state, fringe.actions[:], fringe.cost_backward + cost, 0)
            expanded_fringe.actions.append(action)
            if heuristic is not None:
                expanded_fringe.cost_forward = heuristic(expanded_fringe.state, problem)

            # push
            fringe_queue.push(expanded_fringe)

    return goal_fringe.actions if goal_fringe is not None else []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    return graph_search(problem, priority_func_max_cost_first)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    return graph_search(problem, priority_func_min_cost_first)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    return graph_search(problem, priority_func_min_cost_first)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return graph_search(problem, priority_func_min_cost_first, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
