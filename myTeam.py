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

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


import os # operating system lib
from tensorflow.keras.models import Sequential # , load_model
from tensorflow.keras.layers import Dense
from tensorflow import expand_dims

from tensorflow.keras.saving import save_model
from tensorflow.keras.saving import load_model



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



    first = "BasicAgentAI" 
    second = "BasicAgentAI" # now they are both dumb :/

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

        print()
        print()
        print()
        print()

        print(game_state.__dict__)
        print(dir(game_state))
        print("="*90)
        print(game_state.data.__dict__)
        print(dir(game_state.data))


        width, height = (game_state.data.food.width, game_state.data.food.height)

        print(width, height)

        print(game_state.data.layout.width, game_state.data.layout.height)


        print(game_state.data.__dict__)
        print()
        print(dir(game_state.data))
        print()


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

    r"""
    Garbage text
    C:\Users\Universitat\Documents\GitHub\pacman-agent\pacman-contest\src\contest\
    
    

cd C:\Users\Universitat\Documents\GitHub\pacman-agent
venv\Scripts\activate
cd pacman-contest/src/contest/ 


python capture.py -r agents/IntelArtif_P4_U199140_U185166/myTeam.py -b agents\team_template\myTeam.py

python capture.py -r agents/team_template/myTeam.py -b agents\IntelArtif_P4_U199140_U185166\myTeam.py
python capture.py -r agents/IntelArtif_P4_U199140_U185166/myTeam.py -b agents\IntelArtif-P4_U199140_U185166\myTeam.py


python capture.py -r agents/team_name_2/myTeam.py -b agents/team_name_2/myTeam.py


    python capture.py -r agents/team_template/myTeam.py -b agents/team_name_2/myTeam.py
    python capture.py

    pip install tensorflow
    

    
    """


    return 0

class GameStateSintetizedInfo : 
    # kinda deprecated
    def __init__(self, _current_position, _ally_position, _is_ally_inside_territory, enemy_position, _food, _is_inside_territory, carried_food_quantity, enemy_is_weak) :
        
        # consider adding information from the past? 

        # current agent position [array of ints]
        self.current_position = _current_position 

        # is true if agent is inside of it's territory [bool]
        self.is_inside_territory = _is_inside_territory

        # int for the amount of food the agent is carrying [int]
        self.carried_food = carried_food_quantity

        # position of ally [array of ints]
        self.ally_position = _ally_position

        # is true if ally is inside of it's territory [bool]
        self.is_ally_inside_territory = _is_ally_inside_territory

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

    def get_data_tuple(self) : 

        data = (self.current_position, 
                self.is_inside_territory, 
                self.carried_food, 
                self.ally_position, 
                self.enemy_position_1, 
                self.enemy_1_is_weak, 
                self.enemy_position_2, 
                self.enemy_2_is_weak, 
                self.food)

    def get_info_nn(game_state, capture_agent) : 
        
        team_index = capture_agent.get_team(game_state)
        agent_index = capture_agent.index
        team_index.remove(agent_index) # team_index now should have only 1 value
        game_state.data.agent_states

        agent_position = game_state.data.agent_states[agent_index].get_position()
        data = [x for x in agent_position]

        data.append((agent_position[0] < 16) != capture_agent.red)  # is in territory
        # data.append((agent_position[0] < 16) ^ (capture_agent.red == False)) #is in territory
        
        data.append(0) # food carried

        ally_position = game_state.data.agent_states[team_index[0]].get_position()
        data += [x for x in ally_position]
        data.append((ally_position[0] < 16) !=(capture_agent.red)) #is in territory
        # data.append((ally_position[0] < 16) ^ (capture_agent.red == False)) #is in territory


        enemy_index = capture_agent.get_opponents(game_state)

        enemy_position = game_state.data.agent_states[enemy_index[0]].get_position()
        if(enemy_position == None): 
            enemy_position = [0, 0]
        data += [x for x in enemy_position]
        data.append(0) # is vulnerable

        enemy_position = game_state.data.agent_states[enemy_index[1]].get_position()
        if(enemy_position == None): 
            enemy_position = [0, 0]
        data += [x for x in enemy_position]
        data.append(0) # is vulnerable

        for sub_list in capture_agent.get_food(game_state): 
            data += sub_list

        ret = np.array(data) # 13 + 16 * 32  vals

        return ret


class BasicAgentAI(CaptureAgent) : 
    def __init__(self, _index, time_for_computing=.1): 

        super().__init__(_index, time_for_computing)
        self.start = None
        self.index = _index

        self.training = True
        

        self.full_path = os.path.join("agents", "IntelArtif_P4_U199140_U185166", "model", "tf_pacman_model")

        self.model = load_model(self.full_path) #load model

        self.num_action = 0 # counter for number of actions done


    
    def choose_action(self, game_state): 

        valid_actions = game_state.get_legal_actions(self.index)

        random_factor = 0.33 # 33% chance of random action

        if(random.random() < random_factor) : 
            # do a random valid action

            l = len(valid_actions)
            action = random.choices(valid_actions, weights =[1] + [10] * (l - 1))[0]
            # select random action. Action "stop" is way less likely to be selected 
            # action stop should be the 1st one and always be avaliable

            total_actions = ["Stop", "North", "East", "South",  "West"]

            fake_model_output = np.array([1 if x == action else 0 for x in total_actions])

            self.prev_output = (fake_model_output, action) 

            print(f"Random action ({random_factor * 100}%): {action}\n")

            return action



        prev_state = self.get_previous_observation()

        if(self.training and prev_state != None) : 
            # game_state = self.get_current_observation()

            # expected_result = compute_expected_result(game_state)
            n_epochs = 3
            state_info_nn = GameStateSintetizedInfo.get_info_nn(prev_state, self)

            reward = 0 # TODO: compute reward
            reward  = PacmanRewardFunction.calculate_reward(prev_state, game_state, self.prev_output[1], self.index)

            
            relevance_limit = 0.075
            if not(-relevance_limit < reward < relevance_limit) : 
                # if the reward is interesting enough, train the IA 

                correct_vector = compute_reward_vector(self.prev_output[0], reward)
                
                if True: # debug extra info (?)
                    print(state_info_nn)
                    print(f"Num epochs: {n_epochs}")
                    print(f"expected result: {expected_result}")
                
                expanded_input = expand_dims(state_info_nn, axis=-1)

                self.model.fit(expanded_input, correct_vector, epochs=n_epochs)

            # model.save("tf_pacman_model")

            save_model(model=self.model, filepath=self.full_path, overwrite=True)


        # current_pos = game_state.get_agent_position(self.index)
        # cnd = self.data.layout.walls[0][0]

        total_actions = ["North", "East", "South",  "West"]

        
        state_info_nn = GameStateSintetizedInfo.get_info_nn(game_state, self)
        
        expanded_input = expand_dims(state_info_nn, axis=0)
        # print(f"shape = {expanded_input.shape}")

        # model_output = self.model.predict(expanded_input)  
        model_output = self.model(expanded_input).numpy()[0]


        
        print(f"\nIteration {self.num_action}\t IA results: \t(shape: {model_output.shape})\n{model_output}")


        best_value_action = ("Stop", model_output[0])
        i = 1 
        for action in total_actions : 
            if(action in valid_actions) : 
                if(best_value_action[1] < model_output[i]) : 
                    best_value_action = (action, model_output[i])
            i += 1

        self.prev_output = (model_output, best_value_action[0]) # store record for training

        self.num_action += 1 #update counter
        return best_value_action[0]


    def compute_expected_result(self, game_state):
        # Example: Consider the reward for each action based on the current state
        # You might need to replace this with your actual game logic.

        pacman_position = game_state.get_agent_position(self.index)
        food_positions = game_state.get_food().as_list()

        # Compute distance to the nearest food for each action
        distances_to_food = [manhattan_distance(pacman_position, food) for food in food_positions]

        # Assign a higher reward for actions that move towards the nearest food
        max_distance = max(distances_to_food) if distances_to_food else 1
        normalized_distances = [1 - dist / max_distance for dist in distances_to_food]

        # Create the expected result based on the distances
        expected_result = np.array(normalized_distances)

        return expected_result
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


    def compute_reward_vector(vector_output, reward): 

        """explanation: if we want to rienforce/discourege a behaviour, 
        we can give a positive/negative reward to the agent. The reward "is" the
        derivative of the error function, and the error function is the mse
        (mean square error), therefore: 

        reward = sumatory(correct_vec[i] - vector_output[i])

        assuming vector_output is a list-like and has 5 numbers. Also assuming
        vector_output is is the output of a soft_max

        """

        vector_output = np.array(vector_output[:5]) #convert to np.array, take only 1st 5 elements
        vector_output = vector_output / np.sqrt(np.dot(vector_output, vector_output)) 
        # ^normalize (magnitude = 1)

        if(reward <= 0): #neg reward
            correct_vec = np.array([1, 1, 1, 1, 1]) - vector_output
            # this makes high values low and low values high. 
            # this assumes that the output vector_output is the output of a 
            # softmax. (every val is in [0, 1] and they all sum to 1)
            # this will make np.sum(correct_vect - vector_output) a negative value


        correct_vec = np.square(vector_output) # pronounce optimal choice, discourage others
        correct_vec = correct_vec /np.sqrt(np.dot(correct_vec, correct_vec)) # normalize
        original_reward = np.sum(correct_vec - vector_output) * 0.2
        # 0.2 = 1/5; 5 = len(vector_output)
        
        correct_vec = (reward/original_reward) * correct_vec
        # ^vector with desired reward (I think)

        return correct_vec







class MinimaxAgent(ReflexCaptureAgent):
    # currently unused

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

    # parameters
    bonus_eating_capsule_multiplier = 0.75
    eating_food_reward = 0.5
    penalty_dying = -2.5
    rewand_killing = 2.5
    going_to_centre_reward = 0.15
    moves_with_centre_reward = 20

    def __init__(self):
        # Create an instance of ClassA
        # self.class_a_instance = MinimaxAgent()

        """Instructions: 
        The rewards have to be instantaneus. Giving the IA a positive reward 
        will rienforce all the behaviurs exhivited since the last update. 
        Do not give passive rewards. 

        I marked with TODOs the parts that need atention
        """
    
    def calculate_reward(current_state, next_state, action_taken, agent_index):

        # TODO: add position reward (?)


        # Reward for collecting food
        # food_reward = PacmanRewardFunction.calculate_food_reward(next_state.carried_food - current_state.carried_food)
        food_reward = PacmanRewardFunction.calculate_food_reward(current_state, next_state, agent_index)

        # Reward for killing enemies
        enemy_reward = PacmanRewardFunction.calculate_enemy_reward(current_state, next_state, agent_index)

        going_centre_reward = going_to_centre_reward(current_state, next_state, agent_index)

        # Combine individual rewards (you might want to adjust weights based on importance)
        total_reward = food_reward + enemy_reward + going_centre_reward

        return total_reward

    def calculate_food_reward(current_state, next_state, agent_index): 
        agent_next = next_state.data.agent_states[agent_index]
        new_x, new_y = agent_next.pos

        agent_current = current_state.data.agent_states[agent_index]
        enemy_food_list = agent_current.get_food(current_state) 

        return eating_food_reward * int(enemy_food_list[new_x][new_y]) 

        # return +1 pts if agent just eated a food



    def calculate_enemy_reward(current_state, next_state, agent_index):

        # TODO: return +1.5 pts if agent eated capsule that makes enemies scared
        # +5 pts if agent killed an enemy agent
        # -5 pts if agent was killed

        # eating capsule bonus
        agent_next = next_state.data.agent_states[agent_index]
        new_x, new_y = agent_next.pos

        agent_current = current_state.data.agent_states[agent_index]
        enemy_capsules = agent_current.get_capsules(current_state)

        bonus_eating_capsule = self.bonus_eating_capsule_multiplier * int(enemy_capsules[new_x][new_y])

        # dying penalty

        dying_penalty = 0

        past_x, past_y = agent_current.pos

        displacement_x = new_x - past_x
        displacemet_y = new_y - past_y 

        sq_magnitude = displacement_x * displacement_x + displacement_y * displacement_y

        if(4 < sq_magnitude) : 
            # if agent has "teleported"  more than 2 units of distance, assume it died
            dying_penalty += self.penalty_dying

        # reward killing

        kill_reward = 0

        enemy_list = agent_current.get_opponents(current_state)
        
        # this code may cause errors is the 2 ally agents are very near to each other
        # however, frankly, i don't care
        
        for enemy_index in enemy_list : 
            enemy = current_state.data.agent_states[enemy_index]
            enemy_x, enemy_y = enemy.pos

            dist_to_agent_x = enemy_x - past_x
            dist_to_agent_y = enemy_y - past_y

            dist_to_agent = dist_to_agent_x * dist_to_agent_x + dist_to_agent_y * dist_to_agent_y

            if (4 < dist_to_agent) : continue

            enemy_new = agent_next.data.agent_states[enemy_index]
            new_enemy_x, new_enemy_y = enemy_new.pos

            displ_x = new_enemy_x - enemy_x 
            displ_y = new_enemy_y - enemy_y 

            displ = displ_x * displ_x + displ_y * displ_y

            if (4 < displ) : 
                kill_reward += self.reward_killing
                break # 1 kill per turn

        enemy_reward = bonus_eating_capsule + dying_penalty + kill_reward

        return enemy_reward

    def going_to_centre_reward(current_state, next_state, agent_index) : 
        
        # this is a small reward intendet to make the agents seek the centre of the board
        
        agent_next = next_state.data.agent_states[agent_index]

        if(self.moves_with_centre_reward < agent_next.num_action): 
            return 0

        centre_reward = 0

        agent_current = current_state.data.agent_states[agent_index]
        past_x, past_y = agent_current.pos
        new_x, new_y = agent_next.pos

        if(new_y != past_y): 
            if(abs(16 - new_y) < abs(16 - past_y)): 
                centre_reward += going_to_centre_reward
            else: 
                centre_reward += -going_to_centre_reward/2


        return centre_reward




def reward(self, current_state):
    action_taken = PacmanRewardFunction.getAction(self, current_state)  # Replace with the actual action taken
    next_state =  self.get_successor(self, current_state, action_taken) # Update with the next state after an action


    reward = PacmanRewardFunction.calculate_reward(current_state, next_state, action_taken, self.index)
    print("Reward:", reward)


    
