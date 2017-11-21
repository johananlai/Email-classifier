from __future__ import division 
import numpy as np
from scipy.io import loadmat

def load_data():
    data = loadmat('enron.mat')
    trainFeat = np.array(data['trainFeat'], dtype=bool)
    trainLabels = np.squeeze(data['trainLabels'])
    testFeat = np.array(data['testFeat'], dtype=bool)
    testLabels = np.squeeze(data['testLabels'])
    vocab = np.squeeze(data['vocab'])
    vocab = [vocab[i][0].encode('ascii', 'ignore') for i in xrange(len(vocab))]
    data = dict(trainFeat=trainFeat, trainLabels=trainLabels,
                testFeat=testFeat, testLabels=testLabels, vocab=vocab)
    return data

# Load data
data = load_data()
trainFeat = data['trainFeat']
trainLabels = data['trainLabels']
testFeat = data['testFeat']
testLabels = data['testLabels']
vocab = data['vocab']
W = len(vocab)
'''
    Data description:
    - trainFeat: (Dtrain, W) logical 2d-array of word appearance for training documents.
    - trainLabels: (Dtrain,) 1d-array of {0,1} training labels where 0=ham, 1=spam.
    - testFeat: (Dtest, W) logical 2d-array of word appearance for test documents.
    - testLabels:  (Dtest,) 1d-array of {0,1} test labels where 0=ham, 1=spam.
    - vocab: (W,) 1d-array where vocab[i] is the English characters for word i.
'''

# Different possible vocabularies to use in classification, uncomment chosen line
vocabInds_c = 179  # Filter by word "money"
vocabInds_d = 859  # Filter by word "thanks"
vocabInds_e = 2211  # Filter by word "possibilities"
vocabInds_f = [179, 859, 2211]  # Filter by all 3 words "money", "thanks", and "possibilities"
vocabInds = np.arange(W)  # Use full vocabularly of all W words


# Separate "ham" and "spam" classes, subsample selected vocabulary words
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

trainHam_c = trainFeat[trainLabels == 0][:, vocabInds_c]
trainSpam_c = trainFeat[trainLabels == 1][:, vocabInds_c]

trainHam_d = trainFeat[trainLabels == 0][:, vocabInds_d]
trainSpam_d = trainFeat[trainLabels == 1][:, vocabInds_d]

trainHam_e = trainFeat[trainLabels == 0][:, vocabInds_e]
trainSpam_e = trainFeat[trainLabels == 1][:, vocabInds_e]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
countsSpam = np.sum(trainSpam, axis=0)

countsHam_c = np.sum(trainHam_c, axis=0)
countsSpam_c = np.sum(trainSpam_c, axis=0)

countsHam_d = np.sum(trainHam_d, axis=0)
countsSpam_d = np.sum(trainSpam_d, axis=0)

countsHam_e = np.sum(trainHam_e, axis=0)
countsSpam_e = np.sum(trainSpam_e, axis=0)

# Probabilities calculated from count/num
probContains_givenHam = countsHam/numHam
probContains_givenSpam = countsSpam/numSpam
probNotContains_givenHam = (numHam - countsHam)/numHam
probNotContains_givenSpam = (numSpam - countsSpam)/numSpam

probContains_givenHam_c = countsHam_c/numHam
probContains_givenSpam_c = countsSpam_c/numSpam
probNotContains_givenHam_c = (numHam - countsHam_c)/numHam
probNotContains_givenSpam_c = (numSpam - countsSpam_c)/numSpam

probContains_givenHam_d = countsHam_d/numHam
probContains_givenSpam_d = countsSpam_d/numSpam
probNotContains_givenHam_d = (numHam - countsHam_d)/numHam
probNotContains_givenSpam_d = (numSpam - countsSpam_d)/numSpam

probContains_givenHam_e = countsHam_e/numHam
probContains_givenSpam_e = countsSpam_e/numSpam
probNotContains_givenHam_e = (numHam - countsHam_e)/numHam
probNotContains_givenSpam_e = (numSpam - countsSpam_e)/numSpam

print 'Probability of email containing word "money" given it is ham:', probContains_givenHam_c
print 'Probability of email containing word "money" given it is spam:', probContains_givenSpam_c
print 'Probability of email not containing word "money" given it is ham:', probNotContains_givenHam_c
print 'Probability of email not containing word "money" given it is spam:', probNotContains_givenSpam_c
print
print 'Probability of email containing word "thanks" given it is ham:', probContains_givenHam_d
print 'Probability of email containing word "thanks" given it is spam:', probContains_givenSpam_d
print 'Probability of email not containing word "thanks" given it is ham:', probNotContains_givenHam_d
print 'Probability of email not containing word "thanks" given it is spam:', probNotContains_givenSpam_d
print
print 'Probability of email containing word "possibilities" given it is ham:', probContains_givenHam_e
print 'Probability of email containing word "possibilities" given it is spam:', probContains_givenSpam_e
print 'Probability of email not containing word "possibilities" given it is ham:', probNotContains_givenHam_e
print 'Probability of email not containing word "possibilities" given it is spam:', probNotContains_givenSpam_e
print


# Accuracy of classifiers using probabilities calculated above
M = testLabels.shape[0]

# Classifier using "money"
Yhat_c = []
for email in testFeat:
    if email[vocabInds_c]:
        if probContains_givenSpam_c > probContains_givenHam_c:
            Yhat_c.append(1)
        else:
            Yhat_c.append(0)
    else:
        if probNotContains_givenSpam_c > probNotContains_givenHam_c:
            Yhat_c.append(1)
        else:
            Yhat_c.append(0)

accuracyC = np.sum(testLabels == Yhat_c)/M
print 'Accuracy of classifier based on word "money":', accuracyC

# Classifier using word "thanks"
Yhat_d = []
for email in testFeat:
    if email[vocabInds_d]:
        if probContains_givenSpam_d > probContains_givenHam_d:
            Yhat_d.append(1)
        else:
            Yhat_d.append(0)
    else:
        if probNotContains_givenSpam_d > probNotContains_givenHam_d:
            Yhat_d.append(1)
        else:
            Yhat_d.append(0)
            
accuracyD = np.sum(testLabels == Yhat_d)/M
print 'Accuracy of classifier based on word "thanks":', accuracyD
      
# Classifier using word "possibilities"    
Yhat_e = []
for email in testFeat:
    if email[vocabInds_e]:
        if probContains_givenSpam_e > probContains_givenHam_e:
            Yhat_e.append(1)
        else:
            Yhat_e.append(0)
    else:
        if probNotContains_givenSpam_e > probNotContains_givenHam_e:
            Yhat_e.append(1)
        else:
            Yhat_e.append(0)

accuracyE = np.sum(testLabels == Yhat_e)/M
print 'Accuracy of part 4e classifier based on word "possibilities":', accuracyE

# Classifier using all 3 words
s1 = np.array([probContains_givenSpam_c, probContains_givenSpam_d, probContains_givenSpam_e])
s2 = np.array([probNotContains_givenSpam_c, probNotContains_givenSpam_d, probNotContains_givenSpam_e])
h1 = np.array([probContains_givenHam_c, probContains_givenHam_d, probContains_givenHam_e])
h2 = np.array([probNotContains_givenHam_c, probNotContains_givenHam_d, probNotContains_givenHam_e])

Yhat_f = []
for email in testFeat:
    p1 = np.prod(s1[email[vocabInds_f]]) * np.prod(s2[np.logical_not(email[vocabInds_f])])
    p2 = np.prod(h1[email[vocabInds_f]]) * np.prod(h2[np.logical_not(email[vocabInds_f])])
    
    if p1 > p2:
        Yhat_f.append(1)
    else:
        Yhat_f.append(0)

accuracyF = np.sum(testLabels == Yhat_f)/M
print 'Accuracy of classifier based on words "money", "thanks", and "possibilities":', accuracyF

# Classifier using entire vocabulary

s1 = np.log(probContains_givenSpam)
s2 = np.log(probNotContains_givenSpam)
h1 = np.log(probContains_givenHam)
h2 = np.log(probNotContains_givenHam)

Yhat_g = []
for email in testFeat:
    p1 = np.sum(s1[email[vocabInds]]) + np.sum(s2[np.logical_not(email[vocabInds])])
    p2 = np.sum(h1[email[vocabInds]]) + np.sum(h2[np.logical_not(email[vocabInds])])
    
    if p1 > p2:
        Yhat_g.append(1)
    else:
        Yhat_g.append(0)

accuracyG = np.sum(testLabels == Yhat_g)/M
print 'Accuracy of part 4g classifier based on all words:', accuracyG
print


# Display words that are common in one class, but rare in the other
ind = np.argsort(countsHam-countsSpam)
print 'Words common in Ham but not Spam:'
for i in xrange(-1, -100, -1):
    print vocab[ind[i]],
print
print 'Words common in Spam but not Ham:'
for i in xrange(100):
    print vocab[ind[i]],
