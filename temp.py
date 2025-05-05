import os
import numpy as np

PATH = "/home/adithyaa/KTH/battlesnake/models/PPO"
all_files = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]
ind = [int(f.split("_")[-1].split('.')[0]) for f in all_files]
ind = np.array(ind)
print(ind.argmax())
print(all_files[ind.argmax()])

PATH = PATH + all_files[ind.argmax()]
print(PATH)