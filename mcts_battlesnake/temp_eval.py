from battlesnake_gym.snake_gym import BattlesnakeGym
import time

env = BattlesnakeGym(map_size=(11, 11),
                     number_of_snakes=1)
env.reset()
for i in range(10):
    print(i)
    env.step([3])
    time.sleep(0.5)
    env.render()