import numpy as np

import csv

def readLabels(csvfile):
    scores = []
    genders = []
    races = []
    with open(csvfile) as f:
        linereader = csv.reader(f, delimiter=' ')
        for line in linereader:
            score = float(line[1])

            if line[0][0] == 'm':
                gender = 'male'
            else:
                gender = 'female'

            if line[0][2] == 'w':
                race = 'white'
            else:
                race = 'asian'

            scores.append(score)
            genders.append(gender)
            races.append(race)

    scores = np.array(scores, dtype="float")
#    scores = scores - scores.min()
#    scores = (scores / scores.max()) * 9.0 + 1.0
    genders = np.array(genders)
    races = np.array(races)

    return scores, races, genders

white_male = []
white_female = []
asian_male = []
asian_female = []
train_scores, train_races, train_genders = readLabels('data/train.txt')
for i in range(0, len(train_scores)):
    if train_races[i] == 'white' and train_genders[i] == 'male':
        white_male.append(train_scores)
    if train_races[i] == 'white' and train_genders[i] == 'female':
        white_female.append(train_scores)
    if train_races[i] == 'asian' and train_genders[i] == 'male':
        asian_male.append(train_scores)
    if train_races[i] == 'asian' and train_genders[i] == 'female':
        asian_female.append(train_scores)
white_male = np.array(white_male)
white_female = np.array(white_female)
asian_male = np.array(asian_male)
asian_female = np.array(asian_female)

print('train: ', train_scores.min(), train_scores.max(), train_scores.mean())
print('white-male: ', len(white_male), white_male.min(), white_male.max(), white_male.mean())
print('white-female: ', len(white_female), white_female.min(), white_female.max(), white_female.mean())
print('asian-male: ', len(asian_male), asian_male.min(), asian_male.max(), asian_male.mean())
print('asian-female: ', len(asian_female), asian_female.min(), asian_female.max(), asian_female.mean())

white_male = []
white_female = []
asian_male = []
asian_female = []
validation_scores, validation_races, validation_genders = readLabels('data/test.txt')
for i in range(0, len(validation_scores)):
    if validation_races[i] == 'white' and validation_genders[i] == 'male':
        white_male.append(validation_scores)
    if validation_races[i] == 'white' and validation_genders[i] == 'female':
        white_female.append(validation_scores)
    if validation_races[i] == 'asian' and validation_genders[i] == 'male':
        asian_male.append(validation_scores)
    if validation_races[i] == 'asian' and validation_genders[i] == 'female':
        asian_female.append(validation_scores)
white_male = np.array(white_male)
white_female = np.array(white_female)
asian_male = np.array(asian_male)
asian_female = np.array(asian_female)

print('validation: ', len(validation_scores), validation_scores.min(), validation_scores.max(), validation_scores.mean())
print('white-male: ', len(white_male), white_male.min(), white_male.max(), white_male.mean())
print('white-female: ', len(white_female), white_female.min(), white_female.max(), white_female.mean())
print('asian-male: ', len(asian_male), asian_male.min(), asian_male.max(), asian_male.mean())
print('asian-female: ', len(asian_female), asian_female.min(), asian_female.max(), asian_female.mean())
