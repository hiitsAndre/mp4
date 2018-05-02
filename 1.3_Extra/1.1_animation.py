# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 21:54:32 2018

@author: mac
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import random

#satate = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
state = (0.5, 0.5, 0.03, 0.01, 0.4)
actions = [0, 0.04, -0.04]
rewards = [1, -1, 0]
paddle_x = 1
paddle_height = 0.2
max_times = 10
bx,by,vx,vy,py = 0,1,2,3,4
epsilon = 0.9

def discretize_state(state):
    # In cases where paddle_y = 1 - paddle_height, set discrete_paddle = 11.
    b_x = 11 if state[bx] == 1 else math.floor(state[bx] * 12)
    b_y = 11 if state[by] == 1 else math.floor(state[by] * 12)
    v_x = int(np.sign(state[vx]))
    v_y = 0 if abs(state[vy]) < 0.015 else int(np.sign(state[vy]))
    p_y = 11 if (state[py] == 1 - paddle_height) else math.floor(12 * state[py] / (1 - paddle_height))
    return (b_x, b_y, v_x, v_y, p_y)

def simulate_one_time(state, action):
    #perform action, get current reward and update state
    state[py] += action
    # If the agent tries to move the paddle too high, paddle_y = 0. Likewise, paddle_y = 1 - paddle_height.
    if state[py] < 0:
        state[py] = 0
    if state[py] > 1 - paddle_height:
        state[py] = 1 - paddle_height
    state[bx] += state[vx]
    state[by] += state[vy]
    #If ball_y < 0 (the ball is off the top of the screen), assign ball_y = -ball_y and velocity_y = -velocity_y.
    if state[by] < 0:
        state[by] = -state[by]
        state[vy] = -state[vy]
    #If ball_y > 1 (the ball is off the bottom of the screen), let ball_y = 2 - ball_y and velocity_y = -velocity_y.
    if state[by] > 1:
        state[by] = 2-state[by]
        state[vy] = -state[vy]
    #If ball_x < 0 (the ball is off the left edge of the screen), assign ball_x = -ball_x and velocity_x = -velocity_x.
    if state[bx] < 0:
        state[bx] = -state[bx]
        state[vx] = -state[vx]
    # ball bouncing off the paddle
    bouncing_x = state[bx] >= paddle_x
    bouncing_y = (state[by] >= state[py]) and (state[by] <= state[py] + paddle_height)
    if bouncing_x and bouncing_y:
        state[bx] = 2 * paddle_x-state[bx]
        #make sure that all |velocity_x| > 0.03.
        while True:
            v_x_new = -state[vx] + np.random.uniform(-0.015,0.015)
            if abs(v_x_new)>0.03:
                state[vx] = v_x_new
                break
        state[vy] = state[vy]+ np.random.uniform(-0.03, 0.03)
        return (state,1)
    #Termination
    #as long as ball_x > 1, the game will always be in this state. This is the only state with a reward of -1.
    terminal_x = state[bx] > paddle_x
    terminal_y = (state[by] < state[py]) or (state[by] > state[py] + paddle_height)
    if terminal_x and terminal_y:
        return (None,-1)
    return (state,0)

def select_action(state,Q):
    maxQ = -9e9
    if random.random() >= epsilon: # exploring
        index = random.randint(0,2)
        return actions[index]
    for action in actions:
        key = (discretize_state(state), action)
        if Q[key] > maxQ:
            maxQ = Q[key]
            selected_action = action
    return selected_action

def select_text_action(state,Q):
    maxQ = -9e9
    for action in actions:
        key = (discretize_state(state), action)
        if Q[key] > maxQ:
            maxQ = Q[key]
            selected_action = action
    return selected_action

def max_Q(state,Q):
    #consider terminate state
    if state == None:
        return -1
    maxQ = -9e9
    for action in actions:
        key = (discretize_state(state), action)
        maxQ = max(Q[key],maxQ)
    return maxQ

def Q_learning():
    #Q_key, N_key: discretize_state+action
    Q = {}
    N = {}
    for b_x in range(12):
        for b_y in range(12):
            for v_x in [-1,1]:
                for v_y in [-1,0,1]:
                    for p_y in range(12):
                        for action in actions:
                            Q[((b_x, b_y, v_x, v_y, p_y), action)] = 0
                            N[((b_x, b_y, v_x, v_y, p_y), action)] = 0
    gamma = 0.7
    game_num = 100000
    C = 60
    count = 0
    bounces_sum = 0
    bounces_sum_list = []
    while count < game_num:
        if count % 1000 == 0:
            ave_bounce = bounces_sum / 1000
            bounces_sum_list.append(ave_bounce)
            print('The average number of bounces for 1000 games:',ave_bounce)
            bounces_sum = 0
        current_state = [0.5, 0.5, 0.03, 0.01, 0.4]
        bounce = 0
        while True:
            selected_action = select_action(current_state, Q)
            key = (discretize_state(current_state), selected_action)
            # update Q,N
            N[key] += 1
            next_state, current_reward = simulate_one_time(current_state, selected_action)
            alpha = C / (C + N[key])
            next_state_maxQ = max_Q(next_state, Q)
            Q[key] = Q[key] + alpha * (current_reward + gamma * next_state_maxQ - Q[key])
            #caculate bounce times
            if current_reward == 1:
                bounce += 1
            if current_reward == -1:
                break
            current_state = next_state
        count += 1
        bounces_sum += bounce

    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Mean Episode Rewards')
    plt.title('Mean Episode Rewards vs. Episodes')
    plt.plot(range(int(game_num / 1000)), bounces_sum_list)
    plt.savefig('Mean Episode Rewards.png')

    #test
    
    test_count = 0
    test_num = 200
    bounces_list = []
    testS = []
    while test_count < test_num:
        current_state = [0.5, 0.5, 0.03, 0.01, 0.4]
        bounce = 0
        while True:
            temp = [current_state[0],current_state[1],current_state[2],current_state[3],current_state[4]]
            testS.append(temp)
            selected_action = select_text_action(current_state, Q)
            next_state, current_reward = simulate_one_time(current_state, selected_action)
            #caculate bounce times
            if current_reward == 1:
                bounce += 1
            if current_reward == -1:
                break
            current_state = next_state
        test_count += 1
        bounces_list.append(bounce)
    print(sum(bounces_list)/len(bounces_list))
    return testS

#if __name__=='__main__':
    #start_time = datetime.datetime.now()
    #Q_learning()
    #end_time = datetime.datetime.now()
    #print("The running time:")
    #print(end_time - start_time)
    


import pygame
import sys

pygame.init()

size = width, height = 500, 500
speed = [2, 2]
black = 0, 0, 0

i = 0
testS = Q_learning()

surface = pygame.display.set_mode(size)

run = 1
while run==1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            run = 0
    pygame.time.wait(15)
    surface.fill((0,0,0))      
    pygame.draw.rect(surface,(255,255,255),(0,0,500,5))
    pygame.draw.rect(surface,(255,255,255),(0,495,500,5))
    pygame.draw.rect(surface,(255,255,255),(0,0,5,500))
    pygame.draw.rect(surface,(255,255,255),(495,int(testS[i][4]*500),5,0.2*500))
    pygame.draw.circle(surface,(255,255,255),(int(testS[i][0]*500),int(testS[i][1]*500)),3,0)
    pygame.display.update()
    i = i + 1
    if i == len(testS):
        break
    
pygame.quit()
