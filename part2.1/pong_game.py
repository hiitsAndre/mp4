import numpy as np
import pygame
import os
from pygame.locals import *
from sys import exit
import random
import pygame.surfarray as surfarray
#import matplotlib.pyplot as plt
os.environ['SDL_VIDEODRIVER'] = "dummy"
position = 5, 325
os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
pygame.init()
screen = pygame.display.set_mode((1,1),0,32)

# Creating 2 bars, a ball and background.
back = pygame.Surface((1,1))
background = back.convert()
background.fill((0,0,0))
bar = pygame.Surface((10,50))
paddle = bar.convert()
paddle.fill((0,255,255))
circ_sur = pygame.Surface((15,15))
circ = pygame.draw.circle(circ_sur,(255,255,255),(7,7), (7)  )
circle = circ_sur.convert()
circle.set_colorkey((0,0,0))
font = pygame.font.SysFont("calibri",40)

# Variable initialization.
HIT_REWARD = 1
LOSE_REWARD = -1

class GameState:
    """ The pong game."""
    def __init__(self):
        self.v_threshold = 0.03
        self.paddle_x, self.paddle_y = 1., 0.6
        self.ball_x, self.ball_y = 0.5, 0.5
        self.paddle_move, self.paddle_score = 0,0
        self.paddle_height = 0.2
        self.velocity_x, self.velocity_y = 0.03, 0.01
        self.terminal = 0

    def frame_step(self,input_vect):
        """ Run one step of the game.
        Args:
            input_vect: an array with the actions taken

        Returns:
            image_data: the playground image.
            reward: the reward obtained from the one move in input_vect.
            terminal: the game terminated and the scores are reset.
        """
        # Internally process pygame event handlers.
        pygame.event.pump()

        # Initialize the reward.
        reward = 0

        # Check that only one input action is given.
        if sum(input_vect) != 1:
            raise ValueError('Multiple input actions!')

        # Actions
        if input_vect[1] == 1:
            self.paddle_move = 0
        elif input_vect[2] == 1:
            self.paddle_move = 0.04
        else:
            self.paddle_move = -0.04

        # Scores of the players.
        self.paddle_score = font.render(str(self.paddle_score), True,(255,255,255))

        # Draw the screen.
        screen.blit(background,(0,0))
        frame = pygame.draw.rect(screen,(255,255,255),Rect((5,5),(1,1)),2)
        middle_line = pygame.draw.aaline(screen,(255,255,255),(330,5),(330,475))
        screen.blit(bar1,(self.paddle_x,self.paddle_y))
        screen.blit(circle,(self.circle_x,self.circle_y))
        screen.blit(self.paddle_score,(0.1,0.1))
        #Update the paddle position based on the action chosen by your agent.
        self.paddle_y = self.paddle_y + self.paddle_move
        #Increment ball_x by velocity_x and ball_y by velocity_y.
        self.ball_x = self.ball_x + self.velocity_x
        self.ball_y = self.ball_y + self.velocity_y
        #Bounces
        if self.ball_y < 0.:
            self.ball_y = -self.ball_y
            self.velocity_y = -self.velocity_y
        if self.ball_y > 1.:
            self.ball_y = 2 - self.ball_y
            self.velocity_y = -self.velocity_y
        if self.ball_x < 0.:
            self.ball_x = -self.ball_x
            self.velocity_x = -self.velocity_x
        if self.ball_x > 1.:
            if self.ball_y >= self.paddle_y and self.ball_y <= self.paddle_y + self.paddle_height:
                reward = HIT_REWARD
                self.ball_x = 2*self.paddle_x-self.ball_x
                self.velocity_x = -self.velocity_x + np.random.uniform(-0.015, 0.015)
                if self.velocity_x < 0.03 and self.velocity_x >= 0:
                    self.velocity_x = 0.03
                if self.velocity_x < 0 and self.velocity_x > -0.03:
                    self.velocity_x = -0.03
                self.velocity_y = self.velocity_y + np.random.uniform(-0.03, 0.03)
            else:
                reward = LOSE_REWARD
                terminal = 1

        # Get the playground.
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        pygame.display.update()

        return reward, terminal
