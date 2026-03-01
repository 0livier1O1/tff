import pickle
import numpy as np

with open("/Users/zhgguo/Documents/projects/GreedyTN/results-all-tucker.pickle", "rb") as f:
  result = pickle.load(f)[0][0]
  print(result["greedy-time"])
  for t in result["greedy"][-1]["model"]:
    # put this data into corresponding folder for visualization
    np.save("", t.squeeze().numpy())