import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
   action = env.action_space.sample()  # User-defined policy function
   print(type(action))
   env.render()
   observation, reward, terminated, info = env.step(action)

   if terminated:
      observation = env.reset()
env.close()