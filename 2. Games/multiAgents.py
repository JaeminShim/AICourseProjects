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

    def get_search_index(self, game_state, agent_index, depth):
        return (agent_index - self.index) + (self.depth - depth) * game_state.getNumAgents()

    def get_max_search_index(self, game_state):
        return self.depth * game_state.getNumAgents()

    def get_agent_index_and_depth(self, game_state, index):
        depth = self.depth - index / game_state.getNumAgents()
        agent_index = index % game_state.getNumAgents() - self.index
        if agent_index < 0:
            agent_index = game_state.getNumAgents() - agent_index
            depth -= 1
        return agent_index, depth


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

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.get_alphabeta_score_action(gameState, self.index, self.depth, min_integer, max_integer)[1]

    def get_alphabeta_score_action(self, game_state, agent_index, depth, alpha, beta):
        if depth <= 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state), None
        else:
            return self.get_minimax_score_action(game_state, agent_index, depth, agent_index == self.index, alpha, beta)

    def get_minimax_score_action(self, game_state, agent_index, depth, is_max, alpha, beta):
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        next_depth = depth if next_agent_index != self.index else depth - 1

        result_score = min_integer if is_max else max_integer
        best_action = None

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.get_alphabeta_score_action(successor_state, next_agent_index, next_depth, alpha, beta)[0]
            if result_score < successor_score if is_max else result_score > successor_score:
                result_score = successor_score
                best_action = action

                if is_max:
                    if result_score > beta:
                        break;
                    if result_score > alpha:
                        alpha = result_score
                else:
                    if result_score < alpha:
                        break
                    if result_score < beta:
                        beta = result_score

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
        return self.get_score_action(gameState, 0)[1]

    def get_score_action(self, game_state, index):
        agent_index, depth = self.get_agent_index_and_depth(game_state, index)
        if index >= self.get_max_search_index(game_state) or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state), None
        else:
            func = self.get_max_score_action if agent_index == self.index else self.get_expect_score_action
            return func(game_state, index)

    def get_max_score_action(self, game_state, index):
        agent_index, depth = self.get_agent_index_and_depth(game_state, index)
        result_score = min_integer
        best_action = None

        for action in game_state.getLegalActions(agent_index):
            successor_score = self.get_score_action(game_state.generateSuccessor(agent_index, action), index+1)[0]
            if result_score < successor_score:
                result_score = successor_score
                best_action = action

        return result_score, best_action

    def get_expect_score_action(self, game_state, index):
        agent_index, depth = self.get_agent_index_and_depth(game_state, index)
        result_score = 0.0

        legal_actions = game_state.getLegalActions(agent_index)
        size = len(legal_actions)
        if size > 0:
            for action in legal_actions:
                result_score += self.get_score_action(game_state.generateSuccessor(agent_index, action), index+1)[0]
            result_score /= float(len(legal_actions))

        return result_score, None  # unknown action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    pacman_pos = currentGameState.getPacmanPosition()
    cur_food = currentGameState.getFood()
    closest_food_dist, closest_food_pos = calc_min_food_distance(cur_food, pacman_pos)
    farthest_food_dist = calc_max_food_distance(cur_food, pacman_pos)
    score += 1000 - (farthest_food_dist + 2 * closest_food_dist if cur_food.count() > 1 else closest_food_dist)

    # for i in range(1, currentGameState.getNumAgents()):
    #     ghost_pos = currentGameState.getGhostPosition(i)

    return score

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

