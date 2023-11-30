# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html



import random
import util
import numpy as np

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """

        """
        print(game_state.__dict__)
        print(dir(game_state))
        print("="*90)
        print(game_state.data.__dict__)
        print(dir(game_state.data))
        raise ValueError
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


        """
        Picks among the actions with the highest Q(s,a).
        """
        """
        print(game_state.__dict__)
        print(dir(game_state))
        print("="*90)
        print(game_state.data.__dict__)
        print(dir(game_state.data))
        raise ValueError
        """
        actions = game_state.get_legal_actions(self.index)
        # You can profile your evaluation time by uncommenting these lines
    
    
def fake_function(): 
    # this function is for comments. for some reason, the multiline comments does not work
    # To run (for Marcel): on cmd
        
    # cd C:\Users\Universitat\Documents\GitHub\pacman-agent
    # venv\Scripts\activate
    # cd pacman-contest/src/contest/ 
    
    
    # python capture.py -r agents/team_template/myTeam.py -b agents\IntelArtif-P4_U199140_U185166\myTeam.py
    # python capture.py -r agents/IntelArtif-P4_U199140_U185166/myTeam.py -b agents\team_template\myTeam.py
    # python capture.py -r agents/IntelArtif-P4_U199140_U185166/myTeam.py -b agents\IntelArtif-P4_U199140_U185166\myTeam.py
    
    # python capture.py -r agents/team_template/myTeam.py -b agents/team_name_2/myTeam.py
    # python capture.py
    return 0

class GameStateSintetizedInfo : 
    def __init__(self, _current_position, _ally_position, enemy_position, _food, _is_inside_territory, carried_food_quantity, enemy_is_weak) :
        
        # consider adding information from the past? 

        # current agent position [array of ints]
        self.current_position = _current_position 

        # is true if agent is inside of it's territory [bool]
        self.is_inside_territory = _is_inside_territory

        # int for the amount of food the agent is carrying [int]
        self.carried_food = carried_food_quantity

        # position of ally [array of ints]
        self.ally_position = _ally_position

        # position of the enemy 1 [array of ints]
        self.enemy_position_1 = enemy_position[0]

        # is true if the enemy can be killed by walking into it, false othrewise [bool]
        self.enemy_1_is_weak = enemy_is_weak[0]

        # position of the enemy 2 [array of ints]
        self.enemy_position_2 = enemy_position[1]

        # is true if the enemy can be killed by walking into it, false othrewise [bool]
        self.enemy_2_is_weak = enemy_is_weak[1]
        
        # matrix of 0 and 1, indicating where there is food [array of ints]
        self.food = _food

    def get_data(self) : 
        # join all info in a single array

        data = []
        data = data + self.current_position
        data.append(self.is_inside_territory)
        data.append(self.carried_food)
        data = data + self.ally_position
        data = data + self.enemy_position_1
        data.append(self.enemy_1_is_weak)
        data = data + self.enemy_position_2
        data.append(self.enemy_2_is_weak)
        data = data + self.food
        
        return data





def get_IA_value_function(sintetized_info): 
    return random.random() # random 

class MinimaxAgent(ReflexCaptureAgent):


    def getAction(self, gameState):
        
        numberOfGhosts = gameState.getNumAgents() - 1
        def maxlevel(gameState, depth):
            c_depth = depth + 1
            # we check if the gameState is Win or Lose and if the current depth is equal to the self.depth and if any of this is true 
            # we just return the evaluationFunction 
            if gameState.isWin() or gameState.isLose() or c_depth==self.depth:
                return self.get_IA_value_function(gameState)  # Use your custom evaluation function
            
            maxval = -999999
            legalActions = gameState.getLegalActions(0)
            #iterate through the legalActions
            for action in legalActions:
                #get the succesors and check the max value between itself and the minlevel function using the successor as the gameState
                successor= gameState.generateSuccessor(0, action)
                maxval = max(maxval,minlevel(successor, c_depth, 1))

            return maxval


        def minlevel(gameState, depth,agentIndex):
            minvalue = 999999
            # we check if the gameState is Win or Lose and if any of this is true we just return the evaluationFunction 
            if gameState.isWin() or gameState.isLose():
                return self.get_IA_value_function(gameState)  # Use your custom evaluation function
            
            legalActions = gameState.getLegalActions(agentIndex)
            #iterate through the legalActions
            for action in legalActions:
                successor= gameState.generateSuccessor(agentIndex, action)
                #we check if the current agent is the last one
                if agentIndex == (gameState.getNumAgents() - 1):
                    #if it is the last one, the next agent will be the max
                    minvalue = min(minvalue,maxlevel(successor,depth))
                else:
                    #if it is not the last one, the next agent will also be a min
                    minvalue = min(minvalue,minlevel(successor,depth,agentIndex+1))
            return minvalue


        legalActions = gameState.getLegalActions(0)
        currentScore = -999999
        finalAction = ''
        #iterate through the legalActions
        for action in legalActions:
            next_State = gameState.generateSuccessor(0,action)
            score = minlevel(next_State,0,1)
            #search for the highest score and return the action
            if score > currentScore:
                finalAction = action
                currentScore = score
        return finalAction
    

class PacmanRewardFunction:
    def __init__(self):
        # Create an instance of ClassA
        self.class_a_instance = MinimaxAgent()
    
    def calculate_reward(current_state, next_state, action_taken):

        # Reward for collecting food
        food_reward = PacmanRewardFunction.calculate_food_reward(next_state.carried_food - current_state.carried_food)

        # Reward for killing enemies
        enemy_reward = PacmanRewardFunction.calculate_enemy_reward(next_state.enemy_1_is_weak, next_state.enemy_2_is_weak)

        # Combine individual rewards (you might want to adjust weights based on importance)
        total_reward = food_reward + enemy_reward

        return total_reward

    def calculate_food_reward(food_collected):
        # Define your reward for collecting food
        return food_collected * 10  # You can adjust the multiplier


    def calculate_enemy_reward(enemy_1_is_weak, enemy_2_is_weak):
        # Define your reward for killing enemies
        enemy_reward = 0
        if enemy_1_is_weak:
            enemy_reward += 20  # Adjust rewards based on the importance of killing enemies
        if enemy_2_is_weak:
            enemy_reward += 20

        return enemy_reward

def reward(self, current_state):
    action_taken = PacmanRewardFunction.getAction(self, current_state)  # Replace with the actual action taken
    next_state =  self.get_successor(self, current_state, action_taken) # Update with the next state after an action


    reward = PacmanRewardFunction.calculate_reward(current_state, next_state, action_taken)
    print("Reward:", reward)


    
