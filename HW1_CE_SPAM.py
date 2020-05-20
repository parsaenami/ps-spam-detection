import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
answers = pd.read_csv('answers.csv', header=None)

spamCount = _spamCount = 0
spamProbability = []
_spamProbability = []
ans2 = np.zeros(len(test.get_values()))

trainTranspose = train.T

for s in trainTranspose.get_values()[57]:
    if s == 1:
        spamCount += 1
    else:
        _spamCount += 1

for col in trainTranspose.get_values():
    conditionalProbabilityCount = _conditionalProbabilityCount = 0

    for index in range(len(col)):
        if col[index] == trainTranspose.get_values()[57][index] == 1:
            conditionalProbabilityCount += 1
        elif col[index] == 1 and trainTranspose.get_values()[57][index] == 0:
            _conditionalProbabilityCount += 1

    spamProbability.append(conditionalProbabilityCount / spamCount)
    _spamProbability.append(_conditionalProbabilityCount / _spamCount)

index2 = 0
for row in test.get_values():
    conditionalProbability = _conditionalProbability = 1
    for i in range(len(row)):
        if row[i] == 1:
            conditionalProbability *= spamProbability[i]
            _conditionalProbability *= _spamProbability[i]
        else:
            conditionalProbability *= 1 - spamProbability[i]
            _conditionalProbability *= 1 - _spamProbability[i]

    if conditionalProbability > _conditionalProbability:
        ans2[index2] = 1
    elif conditionalProbability < _conditionalProbability:
        ans2[index2] = 0

    index2 += 1

output = pd.DataFrame(ans2)
output.to_csv("SPM_9531908.csv", header=False, index=False)

compare = pd.concat([output, answers], axis=1)

tmp = pd.DataFrame(compare)
tmp.columns = ["predict", "actual"]

diff = (np.asarray(output) == np.asarray(answers))
accuracy = np.sum(diff) / len(diff)
print ("Accuracy:", accuracy)