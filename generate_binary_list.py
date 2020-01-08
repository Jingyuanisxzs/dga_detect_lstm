import csv
import random

dataPath = './train_data/dga-feed.txt'
resultPath = './train_data/binary_training.txt'

#DGA is marked with 1, whitelist websites are marked with 0
with open(dataPath, "r") as f:
    data = f.read().split('\n')

for i, line in enumerate(data):
    data[i] = line.split(',')[0] + ',1'

with open('./white_list_data/white_list.csv') as whiteList:
    readCSV = csv.reader(whiteList, delimiter=',')
    for row in readCSV:
        if row[1] == 'Domain':
            continue
        data.append(row[1]+',0')
        #there are 847622 samples of DGA so we pick 837622 samples of whitelist sites
        if len(data) >= 847622*2 :
            break

random.shuffle(data)

f_result = open(resultPath, "a")
for a in data:
    f_result.write(a + "\n")