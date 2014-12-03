# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import sys

max_integer = 99999999   # sys.maxint
min_integer = -99999999  # -sys.maxint - 1

from game import Agent

def calc_min_food_distance(foods, position):
    """
    :type foods: Grid
    :type position: tuple
    :rtype: int, tuple

    :param foods: boolean array represents existence of food in map
    :param position: test position
    :return: distance to the closest food
    """
    px, py = position
    if not foods[px][py] and foods.count() > 0:
        sign = [-1, 1]
        max_range = foods.width + foods.height
        for d in range(1, max_range):
            for i in range(d + 1):
                for sign_x in sign:
                    x = px + (i * sign_x)
                    if x < 0 or x >= foods.width:
                        continue

                    for sign_y in sign:
                        y = py + ((d - i) * sign_y)
                        if y < 0 or y >= foods.height:
                            continue

                        if foods[x][y]:
                            return d, (x, y)
    return 0, position

def calc_max_food_distance(foods, position):
    """
    :type foods: Grid
    :type position: tuple
    :rtype: int, tuple

    :param foods: boolean array represents existence of food in map
    :param position: test position
    :return: distance to the farthest food
    """
    px, py = position
    if foods.count() > 0:
        sign = [-1, 1]
        max_range = foods.width + foods.height
        for d in range(max_range - 1, 0, -1):
            for i in range(d + 1):
                for sign_x in sign:
                    x = px + (i * sign_x)
                    if x < 0 or x >= foods.width:
                        continue

                    for sign_y in sign:
                        y = py + ((d - i) * sign_y)
                        if y < 0 or y >= foods.height:
                            continue

                        if foods[x][y]:
                            return d

                        if y == py:
                            break;

                    if x == px:
                        break;
    return 0

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        cur_pos = currentGameState.getPacmanPosition()
        cur_food = currentGameState.getFood()

        score = 0

        for ghostState in newGhostStates:
            ghostDist = manhattanDistance(ghostState.getPosition(), newPos)
            if ghostState.scaredTimer > ghostDist/2:
                return 2000 - ghostDist
            elif ghostDist < 2:
                return -sys.maxint - 1

        closest_food_dist, closest_food_pos = calc_min_food_distance(newFood, newPos)
        farthest_food_dist = calc_max_food_distance(newFood, newPos)
        score += 1000 - (farthest_food_dist + 2 * closest_food_dist if newFood.count() > 1 else closest_food_dist)
        if newPos == cur_pos:
            score = -sys.maxint
        elif cur_food[newPos[0]][newPos[1]]:
            score = sys.maxint

        return score
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.get_minimax_score_action(gameState, self.index, self.depth)[1]

    def get_minimax_score_action(self, game_state, agent_index, depth):
        if depth <= 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state), None
        else:
            func = self.get_max_score_action if agent_index == self.index else self.get_min_score_action
            return func(game_state, agent_index, depth)

    def get_max_score_action(self, game_state, agent_index, depth):
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        next_depth = depth if next_agent_index != self.index else depth - 1

        result_score = min_integer
        best_action = None

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.get_minimax_score_action(successor_state, next_agent_index, next_depth)[0]
            if result_score < successor_score:
                result_score = successor_score
                best_action = action

        return result_score, best_action

    def get_min_score_action(self, game_state, agent_index, depth):
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        next_depth = depth if next_agent_index != self.index else depth - 1

        result_score = max_integer
        best_action = None

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.get_minimax_score_action(successor_state, next_agent_index, next_depth)[0]
            if result_score > successor_score:
                result_score = successor_score
                best_action = action

        return result_score, best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    class PhaseWindow:
        def __init__(self, min_value=min_integer, max_value=max_integer):
            self.min_value = min_value
            self.max_value = max_value

        def is_valid(self, is_max, value):
            if is_max:
                return self.min_value < value
            else:
                return self.max_value > value


        def close(self, is_max, value):
            if is_max:
                if self.max_value > value:
                    self.max_value = value
            else:
                if self.min_value < value:
                    self.min_value = value


        def test_while_min_test(self, min_value):
            return self.min_value < min_value

        def test_while_max_test(self, max_value):
            return self.max_value > max_value

        def clip_min(self, min_value):
            if self.min_value > min_value:
                self.min_value = min_value
                if self.max_value < self.min_value:
                    self.max_value = self.min_value

        def clip_max(self, max_value):
            if self.max_value < max_value:
                self.max_value = max_value
                if self.min_value > self.max_value:
                    self.min_value = self.max_value

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.get_alphabeta_score_action(gameState, self.index, self.depth)[1]

    def get_alphabeta_score_action(self, game_state, agent_index, depth, window=None):
        if depth <= 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state), None
        else:
            func = self.get_max_score_action if agent_index == self.index else self.get_min_score_action
            return func(game_state, agent_index, depth, window)

    def get_max_score_action(self, game_state, agent_index, depth, window):
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        next_depth = depth if next_agent_index != self.index else depth - 1
        # is_next_test_opposite = (agent_index == self.index) != (next_agent_index == self.index)
        # next_window = window if window is not None else self.PhaseWindow() if is_next_test_opposite else None
        next_window = window if window is not None else self.PhaseWindow()

        result_score = min_integer
        best_action = None

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.get_alphabeta_score_action(successor_state,
                                                              next_agent_index,
                                                              next_depth,
                                                              next_window)[0]
            if result_score < successor_score:
                result_score = successor_score
                best_action = action

                if window is not None and not window.is_valid(True, successor_score):
                    best_action = None
                    break

        if best_action is not None and window is not None:
            window.close(True, result_score)

        return result_score, best_action

    def get_min_score_action(self, game_state, agent_index, depth, window):
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        next_depth = depth if next_agent_index != self.index else depth - 1
        # is_next_test_opposite = (agent_index == self.index) != (next_agent_index == self.index)
        # next_window = window if window is not None else self.PhaseWindow() if is_next_test_opposite else None
        next_window = window if window is not None else self.PhaseWindow()

        result_score = max_integer
        best_action = None

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.get_alphabeta_score_action(successor_state,
                                                              next_agent_index,
                                                              next_depth,
                                                              next_window)[0]
            if result_score > successor_score:
                result_score = successor_score
                best_action = action

                if window is not None and not window.is_valid(False, successor_score):
                    best_action = None
                    break

        if best_action is not None and window is not None:
            window.close(False, result_score)
        return result_score, best_action

    def get_minimax_score_action(self, game_state, agent_index, depth, is_max, is_prev_max, window):
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        next_depth = depth if next_agent_index != self.index else depth - 1
        next_window = window if window is not None else self.PhaseWindow()

        result_score = min_integer if is_max else max_integer
        best_action = None

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.get_alphabeta_score_action(successor_state,
                                                              next_agent_index,
                                                              next_depth,
                                                              next_window)[0]
            if result_score < successor_score if is_max else result_score > successor_score:
                result_score = successor_score
                best_action = action

                if window is not None:
                    if is_max != is_prev_max and not window.is_valid(is_max, successor_score):
                        break

        if window is not None and is_max == is_prev_max:
            window.close(is_max, result_score)
        return result_score, best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

