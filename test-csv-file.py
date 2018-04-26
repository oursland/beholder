import numpy as np

import csv

import matplotlib.pyplot as plt

def readLabels(csvfile):
    scores = []
    with open(csvfile) as f:
        linereader = csv.reader(f, delimiter=' ')
        for line in linereader:
            score = float(line[1])
            scores.append(score)

    scores = np.array(scores, dtype="float")
#    scores = scores - scores.min()
#    scores = (scores / scores.max()) * 9.0 + 1.0

    return scores

train = readLabels('data/train.txt')
white_male = readLabels('data/train_wm.txt')
white_female = readLabels('data/train_wf.txt')
asian_male = readLabels('data/train_am.txt')
asian_female = readLabels('data/train_af.txt')

print('train: ', train.min(), train.max(), train.mean(), train.std())
print('white-male: ', len(white_male), white_male.min(), white_male.max(), white_male.mean(), white_male.std())
print('white-female: ', len(white_female), white_female.min(), white_female.max(), white_female.mean(), white_female.std())
print('asian-male: ', len(asian_male), asian_male.min(), asian_male.max(), asian_male.mean(), asian_male.std())
print('asian-female: ', len(asian_female), asian_female.min(), asian_female.max(), asian_female.mean(), asian_female.std())

plt.figure(1, figsize=(12, 8))

plt.subplot(411)
plt.title('Histograms of Rating (Training)')
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.ylabel('White\nMale')
plt.hist(white_male, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(white_male.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(white_male.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(white_male.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

plt.subplot(412)
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.ylabel('White\nFemale')
plt.hist(white_female, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(white_female.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(white_female.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(white_female.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

plt.subplot(413)
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.ylabel('Asian\nMale')
plt.hist(asian_male, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(asian_male.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(asian_male.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(asian_male.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

plt.subplot(414)
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.xlabel('Rating')
plt.ylabel('Asian\nFemale')
plt.hist(asian_female, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(asian_female.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(asian_female.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(asian_female.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

test = readLabels('data/test.txt')
white_male = readLabels('data/test_wm.txt')
white_female = readLabels('data/test_wf.txt')
asian_male = readLabels('data/test_am.txt')
asian_female = readLabels('data/test_af.txt')

print('test: ', test.min(), test.max(), test.mean(), test.std())
print('white-male: ', len(white_male), white_male.min(), white_male.max(), white_male.mean(), white_male.std())
print('white-female: ', len(white_female), white_female.min(), white_female.max(), white_female.mean(), white_female.std())
print('asian-male: ', len(asian_male), asian_male.min(), asian_male.max(), asian_male.mean(), asian_male.std())
print('asian-female: ', len(asian_female), asian_female.min(), asian_female.max(), asian_female.mean(), asian_female.std())

plt.figure(2, figsize=(12, 8))

plt.subplot(411)
plt.title('Histograms of Rating (Testing)')
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.ylabel('White\nMale')
plt.hist(white_male, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(white_male.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(white_male.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(white_male.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

plt.subplot(412)
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.ylabel('White\nFemale')
plt.hist(white_female, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(white_female.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(white_female.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(white_female.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

plt.subplot(413)
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.ylabel('Asian\nMale')
plt.hist(asian_male, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(asian_male.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(asian_male.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(asian_male.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

plt.subplot(414)
plt.axis([0, 5, 0, 1])
plt.grid(True)
plt.xlabel('Rating')
plt.ylabel('Asian\nFemale')
plt.hist(asian_female, 50, density=1, facecolor='purple', alpha=0.60)
plt.axvline(asian_female.mean(), color='k', linestyle='solid', linewidth=4)
plt.axvline(asian_female.mean() + white_male.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(asian_female.mean() - white_male.std(), color='k', linestyle='dashed', linewidth=2)

plt.show()
