import numpy as np
import math
import warnings
from collections import Counter
import pandas as pd
import random
from random import randrange
import copy

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    #print(Counter(votes).most_common(1))
    #print(confidence)
    return vote_result

def get_cuisine(cuisine):
    if cuisine == "brazilian":
        return 0
    elif cuisine == "british":
        return 1
    elif cuisine == "cajun_creole":
        return 2
    elif cuisine == "chinese":
        return 3
    elif cuisine == "filipino":
        return 4
    elif cuisine == "french":
        return 5
    elif cuisine == "greek":
        return 6
    elif cuisine == "indian":
        return 7
    elif cuisine == "irish":
        return 8
    elif cuisine == "italian":
        return 9
    elif cuisine == "jamaican":
        return 10
    elif cuisine == "japanese":
        return 11
    elif cuisine == "korean":
        return 12
    elif cuisine == "mexican":
        return 13
    elif cuisine == "moroccan":
        return 14
    elif cuisine == "russian":
        return 15
    elif cuisine == "southern_us":
        return 16
    elif cuisine == "spanish":
        return 17
    elif cuisine == "thai":
        return 18
    elif cuisine == "vietnamese":
        return 19
    else:
        return -1

## Get all Ingredients
print("Parsing ingredients (features)...")
ing = pd.read_csv("ingredients.txt", header=None)
ing = ing.values.tolist()
print("Done.")

## Get Training Set
print("Preparing DataSet...")
df = pd.DataFrame()
with open("training.csv", 'r') as f:
    for line in f:
        df = pd.concat( [df, pd.DataFrame([tuple(line.strip().split(','))])], ignore_index=True )

## Drop ID Column
df.drop(0, 1, inplace = True)

## Data to Array
full_data = df.values.tolist()

#print(len(full_data))

print("Done.")
## Get rid of double quotes
count = 0
for row in full_data:
    modified = []
    for data in row:
        if type(data) is str:
            modified.append(data[1:-1])
    full_data[count] = modified
    count += 1
        
rowcount = 0
tmpperc = -1
for row in full_data:
    cuisine = row[0]
    new_row = copy.deepcopy(ing)
    datacount = 0
    for ingredient in new_row:
        if ingredient[0] in row:
            new_row[datacount] = 1
            datacount += 1
        else:
            new_row[datacount] = 0
            datacount += 1
    full_data[rowcount] = new_row
    full_data[rowcount].append(get_cuisine(cuisine))
    rowcount += 1
    percentage = int((rowcount+1)/1794*100)
    if not tmpperc == percentage:
        print('{}{}{}'.format("Parsing dataset... ", int((rowcount+1)/1794*100), "% \r"), end='')
    tmpperc = percentage

print("")
random.shuffle(full_data)

def cross_validation_split(dataset, folds=6):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

folds = cross_validation_split(full_data)

print("Running cross validation...")
for run in range(6):
    print("Validating with Fold", run+1, "...")
    train_data = []
    test_data = folds[run]
    train_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}
    test_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}

    for i in range(6):
        if not i == run: 
            train_data += folds[run]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote = k_nearest_neighbors(train_set, data, k=35)
            if group == vote:
                correct += 1
            total += 1
    print(30*"-")
    print('Accuracy: ', correct/total)
    print('Generalization Error: ', 1/len(test_data) * (total-correct))
    print(30*'-')

##inp = ""
##query = []
##print("Do you want to test an input against the whole training set? \n If yes, start entering ingredients and type \"done\" when finished. \n If not, just type \"done\"")
##while not inp == "done":
##    inp = input("Enter ingredient: ")
##    if inp == "done":
##          break
##    else:
##          query.append(inp)
##          print(query)
##
##if not len(query) == 0:
##    new_row = copy.deepcopy(ing)
##    datacount = 0
##    for ingredient in new_row:
##        if ingredient[0] in query:
##            new_row[datacount] = 1
##            datacount += 1
##        else:
##            new_row[datacount] = 0
##            datacount += 1
##    new_row.append(get_cuisine("brazilian"))
##    train_data = full_data
##    test_data = [new_row]
##    print(test_data)
##    train_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}
##    test_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}
##    for i in train_data:
##        train_set[i[-1]].append(i[:-1])
##    for i in test_data:
##        test_set[i[-1]].append(i[:-1])
##    for group in test_set:
##        for data in test_set[group]:
##            print("I guess the cuisine is...", k_nearest_neighbors(train_set, data, k=49))
