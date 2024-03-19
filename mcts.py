# myTeam.py
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
#
# Written by chen chen s4139399
# Date: 14/03/2024


from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions, GameStateData
from capture import GameState
import game

import numpy as np
import math


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

###########################
# Structure for MCTS Node #
###########################


class MCTSNode(object):

    def __init__(self, gameState, agent, action, parent, enemy_position, borderline):

        self.parent = parent
        self.action = action
        self.child = []
        self.visit_times = 1
        self.value = 0.0
        self.depth = parent.depth + 1 if parent else 0
        self.gameState = gameState.deepCopy()
        self.enemy_position = enemy_position
        self.legalActions = [act for act in gameState.getLegalActions(
            agent.index) if act != 'Stop']
        self.unexploredActions = self.legalActions[:]
        self.borderline = borderline
        self.agent = agent
        # self.epsilon = 1
        # self.rewards = 0

    # Selection
    def select_best_child(self, exploration_constant=2):
        best_score = -np.inf
        best_child = None

        for candidate in self.child:
            # UCB1
            exploitation_term = candidate.value / candidate.visit_times
            exploration_term = math.sqrt(
                exploration_constant * math.log(self.visit_times) / candidate.visit_times)
            score = exploitation_term + exploration_term
            if score > best_score:
                best_child = candidate
                best_score = score

        return best_child

    # Expansion
    def node_expansion(self):
        max_depth = 15
        if self.depth >= max_depth:
            return self

        if self.unexploredActions != []:
            action = self.unexploredActions.pop()
            cucrrent_state = self.gameState.deepCopy()
            next_state = cucrrent_state.generateSuccessor(
                self.agent.index, action)
            child_node = MCTSNode(
                next_state, self.agent, action, self, self.enemy_position, self.borderline)
            self.child.append(child_node)
            return child_node

        # if util.flipCoin(self.epsilon):
        # else:
        #     next_best_child = random.choice(self.child)
        next_best_child = self.select_best_child()

        return next_best_child.node_expansion()

    # Simulation
    def evaluate(self):
        current_position = self.gameState.getAgentPosition(self.agent.index)
        if current_position == self.gameState.getInitialAgentPosition(self.agent.index):
            return -1000
        value = self.get_features() * MCTSNode.get_weights(self)
        return value

    def get_features(self):
        feature = util.Counter()
        current_position = self.gameState.getAgentPosition(self.agent.index)
        feature['distance'] = min([self.agent.getMazeDistance(
            current_position, border_position) for border_position in self.borderline])
        return feature

    def get_weights(self):
        return {'distance': -1}

    # Back-propagation
    def back_propagation(self, reward):
        self.visit_times += 1
        self.value += reward

        if self.parent is not None:
            self.parent.back_propagation(reward)

    def mcts_search(self):
        time_limit = 0.99
        start = time.time()

        while (time.time() - start < time_limit):
            selected_node = self.node_expansion()
            reward = selected_node.evaluate()
            selected_node.back_propagation(reward)
        print("mcts_search")

        return self.select_best_child().action


##########
# Agents #
##########


class BaseAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        self.arena_width = gameState.data.layout.width
        self.arena_height = gameState.data.layout.height
        self.my_border = self.get_my_border(gameState)
        self.enemy_border = self.get_enemy_border(gameState)
        self.food_count = int(len(self.getFood(gameState).asList()))

    def get_my_border(self, gameState):
        # my border position, used as the parameter of MCTS node.
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2 - 1
        else:
            border_x = self.arena_width // 2
        border_line = [(border_x, h) for h in range(self.arena_height)]

        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2 * self.red, y) not in walls]

    def get_enemy_border(self, gameState):
        # enemy border position, used as the parameter of MCTS node.
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2
        else:
            border_x = self.arena_width // 2 - 1
        border_line = [(border_x, h) for h in range(self.arena_height)]

        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2 * self.red, y) not in walls]

    def detect_enemy_pacman(self, gameState):
        # return a list of positions of enemy's pacman
        enemy_pacman_list = []

        for enemy in self.getOpponents(gameState):
            enemy_state = gameState.getAgentState(enemy)
            if enemy_state.isPacman and gameState.getAgentPosition(enemy) != None:
                enemy_pacman_list.append(enemy)

        return enemy_pacman_list

    def detect_enemy_ghost(self, gameState):
        # return a list of positions of enemy's ghost
        enemy_ghost_list = []

        for enemy in self.getOpponents(gameState):
            enemy_state = gameState.getAgentState(enemy)
            if (not enemy_state.isPacman) and enemy_state.scaredTimer == 0:
                enemy_position = gameState.getAgentPosition(enemy)
                if enemy_position != None:
                    enemy_ghost_list.append(enemy)

        return enemy_ghost_list

    def detect_ghost_nearby(self, gameState, n_steps=5):
        # return a list of positions of enemy's ghost within 5 steps
        ghost_nearby = []
        my_position = gameState.getAgentPosition(self.index)
        ghosts = self.detect_enemy_ghost(gameState)

        for ghost in ghosts:
            distance = self.getMazeDistance(
                my_position, gameState.getAgentPosition(ghost))
            if distance <= n_steps:
                ghost_nearby.append(ghost)

        return ghost_nearby

    def get_min_distance_to_food(self, gameState):
        my_position = gameState.getAgentPosition(self.index)
        return min([self.getMazeDistance(my_position, food) for food in self.getFood(gameState).asList()])

    def get_min_distance_to_capsule(self, gameState):
        my_position = gameState.getAgentPosition(self.index)
        return min([self.getMazeDistance(my_position, capsule) for capsule in self.getCapsules(gameState)])

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        num_carrying = agent_state.numCarrying
        is_pacman = agent_state.isPacman

        if is_pacman:
            print("1")
            ghost_nearby_position = [gameState.getAgentPosition(
                ghost) for ghost in self.detect_ghost_nearby(gameState)]
            food_list = self.getFood(gameState).asList()

            if not ghost_nearby_position:
                values = [self.evaluate_off(gameState, action)
                          for action in actions]
                max_value = max(values)
                best_actions = [action for action, value in zip(
                    actions, values) if value == max_value]
                action_chosen = random.choice(best_actions)
            elif len(food_list) < 2 or num_carrying > 7:
                rootNode = MCTSNode(gameState, self, None,
                                    None, ghost_nearby_position, self.my_border)
                action_chosen = MCTSNode.mcts_search(rootNode)
                print("2")
            else:
                rootNode = MCTSNode(gameState, self, None,
                                    None, ghost_nearby_position, self.my_border)
                action_chosen = MCTSNode.mcts_search(rootNode)
                print("3")

        else:
            print("4")
            ghosts = self.detect_enemy_ghost(gameState)
            values = [self.evaluate_def(gameState, action)
                      for action in actions]
            max_value = max(values)
            best_actions = [action for action, value in zip(
                actions, values) if value == max_value]
            action_chosen = random.choice(best_actions)

        return action_chosen

    def evaluate_off(self, gameState, action):
        features = self.get_off_features(gameState, action)
        weights = self.get_off_weights(gameState, action)

        return features * weights

    def get_off_features(self, gameState, action):

        pass

    def get_off_weights(self, gameState, action):

        pass

    def evaluate_def(self, gameState, action):
        features = self.get_def_features(gameState, action)
        weights = self.get_def_weights(gameState, action)

        return features * weights

    def get_def_features(self, gameState, action):

        pass

    def get_def_weights(self, gameState, action):

        pass


class OffensiveAgent(BaseAgent):

    def get_off_features(self, gameState, action):
        features = util.Counter()
        next_state = gameState.generateSuccessor(self.index, action)

        if next_state.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1

        if len(self.getFood(next_state).asList()) > 0:
            features['minDistanceToFood'] = self.get_min_distance_to_food(
                next_state)

        if self.getCapsules(gameState) != None:
            features['getCapsules'] = self.get_min_distance_to_capsule(
                next_state)

        return features

    def get_off_weights(self, gameState, action):

        return {'minDistanceToFood': -1, 'getCapsules': 1000, 'getFood': 100}

    def get_def_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            current_pos = successor.getAgentState(self.index).getPosition()
            min_distance = min([self.getMazeDistance(
                current_pos, food) for food in foodList])
            features['distanceToFood'] = min_distance
        return features

    def get_def_weights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveAgent(BaseAgent):

    def get_def_features(self, gameState, action):
        features = util.Counter()
        next_state = gameState.generateSuccessor(self.index, action)

        my_state = next_state.getAgentState(self.index)
        my_position = my_state.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if my_state.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [next_state.getAgentState(i)
                   for i in self.getOpponents(next_state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                my_position, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_def_weights(self, gameState, action):

        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
