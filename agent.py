import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAi, Direction, Point
from model import Linear_QNet, Qtrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
RANDOM_MOVE = 80
HIDDEN_SIZE = 256


class Agent:

    def __init__(self):
        self.num_games = 0
        self.epsilon = 0   # parameter to control randomness
        self.gamma = 0.9   # discount rate / must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)  # if we exceed memory popleft() called to remove and clear memory
        self.model = Linear_QNet(11, HIDDEN_SIZE, 3)  # 11 State input and output for action needs to be 3
        self.trainer = Qtrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)
        # model, Trainer TODO

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        direction_l = game.direction == Direction.LEFT
        direction_r = game.direction == Direction.RIGHT
        direction_u = game.direction == Direction.UP
        direction_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (direction_r and game.is_collision(point_r)) or
            (direction_l and game.is_collision(point_l)) or
            (direction_u and game.is_collision(point_u)) or
            (direction_d and game.is_collision(point_d)),

            # Danger right
            (direction_u and game.is_collision(point_u)) or
            (direction_d and game.is_collision(point_d)) or
            (direction_l and game.is_collision(point_l)) or
            (direction_r and game.is_collision(point_r)),

            # Danger left
            (direction_d and game.is_collision(point_d)) or
            (direction_u and game.is_collision(point_u)) or
            (direction_r and game.is_collision(point_r)) or
            (direction_l and game.is_collision(point_l)),

            # Move direction
            direction_l,
            direction_r,
            direction_u,
            direction_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # popleft if Max_Memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            minibatch = random.sample(self.memory, BATCH_SIZE)   # list of tuples
        else:
            minibatch = self.memory

        states, actions, rewards, next_states, game_overs = zip(*minibatch)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        # for state, action, reward, next_state, game_over in minibatch:
        #       self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = RANDOM_MOVE - self.num_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_score = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAi()
    while True:
        # get old/current state
        state_old = agent.get_state(game)

        # get move based on current state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)  # get new state

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember training
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_model()

            print('Game', agent.num_games, 'Score', score, 'Record', record)

            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_score, plot_mean_scores)


if __name__ == '__main__':
    train()
