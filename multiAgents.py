# multiAgents.py
# --------------
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

# Authors: Ayesha Kazi, Mahina Khan

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = successorGameState.getScore()

        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 15.0 / (minFoodDist + 1)

        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            if newScaredTimes[i] > 0:
                score += 30.0 / (ghostDist + 1)
            else:
                if ghostDist < 2:
                    score -= 300 
                else:
                    score -= 5.0 / (ghostDist + 1)

        if action == Directions.STOP:
            score -= 15

        return score
        

def scoreEvaluationFunction(currentGameState: GameState):
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
    
    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        
        def value(state, agentIndex, depthLeft):
            """
            Recursive minimax value.
            - state: GameState at this node
            - agentIndex: whose turn (0 = Pacman, 1..N-1 = ghosts)
            - depthLeft: remaining plies (Pacman+all ghosts)
            """
            
            if depthLeft == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                best = float('-inf')
                for a in actions:
                    success = state.generateSuccessor(0, a)
                    best = max(best, value(success, 1, depthLeft))
                return best

            else:
                best = float('inf')
                if agentIndex < numAgents - 1:
                    nextAgent = agentIndex + 1
                    nextDepthLeft = depthLeft
                else:
                    nextAgent = 0
                    nextDepthLeft = depthLeft - 1

                for a in actions:
                    success = state.generateSuccessor(agentIndex, a)
                    best = min(best, value(success, nextAgent, nextDepthLeft))
                return best
        actions = gameState.getLegalActions(0)    
        bestVal = float('-inf')
        bestAction = None
        for a in actions:
            succ = gameState.generateSuccessor(0, a)
            v = value(succ, 1, self.depth)  
            if v > bestVal:
                bestVal = v
                bestAction = a

        return bestAction
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depthLeft, alpha, beta):
            """
            Recursive alpha-beta value.
            """
            if depthLeft == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            # max
            if agentIndex == 0:
                v = float('-inf')
                for a in actions:
                    succ = state.generateSuccessor(0, a)
                    v = max(v, value(succ, 1, depthLeft, alpha, beta))
                    # strict pruning
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            # min
            else:
                v = float('inf')
                # Determine next (agentIndex, depthLeft)
                if agentIndex < numAgents - 1:
                    nextAgent = agentIndex + 1
                    nextDepthLeft = depthLeft
                else:
                    nextAgent = 0
                    nextDepthLeft = depthLeft - 1

                for a in actions:  
                    succ = state.generateSuccessor(agentIndex, a)
                    v = min(v, value(succ, nextAgent, nextDepthLeft, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        actions = gameState.getLegalActions(0)

        bestVal = float('-inf')
        bestAction = None
        alpha, beta = float('-inf'), float('inf')

        for a in actions:
            succ = gameState.generateSuccessor(0, a)
            v = value(succ, 1, self.depth, alpha, beta)  
            if v > bestVal:
                bestVal = v
                bestAction = a
            if bestVal > beta:   
                break
            alpha = max(alpha, bestVal)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depthLeft):
            """
            Recursive expectimax value.
            """
            if depthLeft == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            # max
            if agentIndex == 0:
                best = float('-inf')
                for a in actions:  
                    succ = state.generateSuccessor(0, a)
                    best = max(best, value(succ, 1, depthLeft))
                return best
            # ghosts modeled as choosing uniformly at random
            else:
                if agentIndex < numAgents - 1:
                    nextAgent = agentIndex + 1
                    nextDepthLeft = depthLeft
                else:
                    nextAgent = 0
                    nextDepthLeft = depthLeft - 1

                prob = 1.0 / len(actions)  
                expected = 0.0
                for a in actions:  
                    succ = state.generateSuccessor(agentIndex, a)
                    expected += prob * value(succ, nextAgent, nextDepthLeft)
                return expected

        actions = gameState.getLegalActions(0)
        if not actions:
            from game import Directions
            return Directions.STOP

        bestVal = float('-inf')
        bestAction = None
        for a in actions:  
            succ = gameState.generateSuccessor(0, a)
            v = value(succ, 1, self.depth)
            if v > bestVal:
                bestVal = v
                bestAction = a
        return bestAction
       

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Pacman's game score is combined with weighted heuristics 
    which include the food, capsules and ghost distance. Closeness to food and scared ghosts is rewarded
    while active nearby ghosts are penalized.
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    total = currentGameState.getScore()

    if foodList:
        nearestFood = min(manhattanDistance(pacPos, f) for f in foodList)
        total += 17.3 / (nearestFood + 1.2)    
        total -= 3.7 * len(foodList)           

    if capsules:
        nearestCapsule = min(manhattanDistance(pacPos, c) for c in capsules)
        total += 11.9 / (nearestCapsule + 1.1)
        total -= 18.4 * len(capsules)

    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacPos, ghostPos)

        if ghost.scaredTimer > 0:
            total += 33.7 / (ghostDist + 1.3)
        else:
            if ghostDist < 2:
                total -= 426.5                 
            else:
                total -= 4.6 / (ghostDist + 0.9)

    if foodList:
        avgFoodDist = sum(manhattanDistance(pacPos, f) for f in foodList) / len(foodList)
        total += 6.8 / (avgFoodDist + 2.0)

    return total

# Abbreviation
better = betterEvaluationFunction
