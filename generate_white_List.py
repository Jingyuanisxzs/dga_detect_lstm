import csv

with open('example.csv') as whiteList:
    readCSV = csv.reader(whiteList, delimiter=',')
    for row in readCSV:
        print(row[1])