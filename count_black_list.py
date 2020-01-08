dataPath = './train_data/dga-feed.txt'

with open(dataPath, "r") as f:
    data = f.read().split('\n')

print(len(data))