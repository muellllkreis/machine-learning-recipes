import numpy as np
import math
import warnings
from collections import Counter
import pandas as pd
import random
from random import randrange
import copy
from scipy.spatial.distance import cdist

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
        #euclidean_distance = cdist(group, predict, metric='cosine')

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
#print(ing)

## Get Training Set
print("Preparing DataSet...")
df = pd.DataFrame()
with open("training.csv", 'r') as f:
    for line in f:
        df = pd.concat( [df, pd.DataFrame([tuple(line.strip().split(','))])], ignore_index=True )

#df.replace('what to replace', what value, inplace = true)

## Drop ID Column
df.drop(0, 1, inplace = True)

#print(df.ix[254])
#print(df.ix[100])
#print(df.head)

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
        
print(full_data[0])

rowcount = 0
tmpperc = -1
for row in full_data:
    cuisine = row[0]
##    print(row[0])
##    print(row[1])
    new_row = copy.deepcopy(ing)
    datacount = 0
    for ingredient in new_row:
        if ingredient[0] in row:
            #print("here")
            new_row[datacount] = 1
            datacount += 1
        else:
            new_row[datacount] = 0
            datacount += 1
    #full_data[rowcount] = np.append(new_row, get_cuisine(cuisine))
    full_data[rowcount] = new_row
    full_data[rowcount].append(get_cuisine(cuisine))
    rowcount += 1
    ######################## DELETE THIS FOR FULL DATA
    ##if(rowcount == 4):
    ##    break;
    ###################################################
    percentage = int((rowcount+1)/1794*100)
    if not tmpperc == percentage:
        print('{}{}{}'.format("Parsing dataset... ", int((rowcount+1)/1794*100), "% \r"))
    tmpperc = percentage

##
###print(full_data)
##for i in full_data[0]:
##    print(i)
###print(full_data[0])
##print("###########################################################################")
##for i in full_data[1]:
##    print(i)

random.shuffle(full_data)

#print(full_data[:1])

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
##for i in folds:
##    print(len(i))

print("Running cross validation...")
for run in range(6):
    print("Validating with Fold", run+1)
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
    print('Accuracy: ', correct/total)
    print('Generalization Error: ', 1/len(test_data) * (total-correct))
    print(30*'#')

    
##test_size = 0.2
##train_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}
##test_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}
##train_data = full_data[:-int(test_size*len(full_data))]
##test_data = full_data[-int(test_size*len(full_data)):]
##
##for i in train_data:
##    train_set[i[-1]].append(i[:-1])
##    
##for i in test_data:
##    test_set[i[-1]].append(i[:-1])
##
##correct = 0
##total = 0
##
##print("Classifying...")
##for group in test_set:
##    for data in test_set[group]:
##        vote = k_nearest_neighbors(train_set, data, k=35)
##        if group == vote:
##            correct += 1
##        total += 1
##print('Accuracy: ', correct/total)
##
##
##
##
##
##
##
###print(full_data[1])
#print(full_data[2])
##new_row = copy.deepcopy(ing)
##count = 0
##
##modified = []
##for data in full_data[0]:
##    print(data)
##    if type(data) is str:
##        modified.append(data[1:-1])
##    count += 1
##
##count = 0
##
##print(modified)
##for ingredient in new_row:
##    if ingredient in modified:
##        new_row[count] = 1
##        count += 1
##    else:
##        new_row[count] = 0
##        count += 1
##
##print(ing)
