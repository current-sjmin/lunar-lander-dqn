import pygame
import gym

env = gym.make("LunarLander-v2", render_mode='human')
pygame.init()

while True:
    state, data_type = env.reset()
    done = False
    total_reward = 0
    while not done:
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_LEFT]:
            action = 1
        if keys[pygame.K_UP]:
            action = 2
        if keys[pygame.K_RIGHT]:
            action = 3
        next_state, reward, done, truncated, info = env.step(action=action)
        done = done or truncated
        total_reward += reward
        state = next_state
    print(f"total reward : {total_reward}")
