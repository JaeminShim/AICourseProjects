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

max_integer = sys.maxint
min_integer = -sys.maxint - 1

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
        return self.get_action(gameState, 0, self.depth)


    def get_action(self, game_state, agent_index, depth):
        is_pacman = agent_index == 0
        agents_size = game_state.getNumAgents()

        if depth <= 0:
            return Directions.STOP

        next_agent = (agent_index+1) % agents_size
        next_depth = depth - 1 if next_agent == 0 else depth

        try:
            func = self.get_min_score_action
            if is_pacman:
                func = self.get_max_score_action
            return func(game_state.generateSuccessor(agent_index, self.get_action(game_state, next_agent, next_depth)),
                        agent_index)
        except:
            return Directions.STOP

    def get_max_score_action(self, game_state, agent_index):
        result_score = self.evaluationFunction(game_state)
        best_action = Directions.STOP

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.evaluationFunction(successor_state)
            if result_score < successor_score:
                result_score = successor_score
                best_action = action

        return best_action

    def get_min_score_action(self, game_state, agent_index):
        result_score = self.evaluationFunction(game_state)
        best_action = Directions.STOP

        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            successor_score = self.evaluationFunction(successor_state)
            if result_score > successor_score:
                result_score = successor_score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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

