import numpy as np
import matplotlib.pyplot as plt
np.random.seed(57)      

def getFeatureDim(dataset):
    mx = 0
    linesplit1 = dataset.split('\n')
    for line2 in linesplit1:
        spacesplit2 = line2.split()
        if spacesplit2 != []:
            for i2 in range(1,len(spacesplit2)):
                colonsplit2 = spacesplit2[i2].split(':')
                ind = int(colonsplit2[0])                
                if ind >= mx:
                    mx = ind
    return mx

def libsvm2matrix(dataset, featuresize):    
    newdataset = []
    nwlinesplit = dataset.split('\n')    
    for line1 in nwlinesplit:
        spacesplit = line1.split()        
        nwline1 = [0]*(featuresize+1)
        if spacesplit != []:
            nwline1[0] = spacesplit[0]
            for i1 in range(1,len(spacesplit)):
                colonsplit = spacesplit[i1].split(':')
                ind = int(colonsplit[0])                
                nwline1[ind] = float(colonsplit[-1])        
            newdataset.append(nwline1)
    return newdataset

def perceptron(dataset, weight, bias, eta, mu):
    for line2 in dataset:        
        y = int(line2[0])        
        factor = y * (np.inner(line2[1:],weight)+bias)        
        if factor < mu:
            x = np.asarray(line2[1:])
            weight = weight + (eta * y) * x
            bias = bias + (eta * y)            
    bias = np.asarray(bias)    
    return np.append(bias,weight)

def updatecount(dataset, weight, bias, eta, mu):
    update = 0
    for line2 in dataset:        
        y = int(line2[0])        
        factor = y * (np.inner(line2[1:],weight)+bias)        
        if factor < mu:
            update += 1
            x = np.asarray(line2[1:])
            weight = weight + (eta * y) * x
            bias = bias + (eta * y)            
    bias = np.asarray(bias)    
    return update

def accuracyCalc(dataset,weight, featuresize):
    correctcount = 0
    total = 0        
    testdata = libsvm2matrix(dataset, featuresize)
    bias = np.array(weight[0])
    weight = np.array(weight[1:])
    for line3 in testdata:
        total += 1        
        y = int(line3[0])
        x = np.array(line3[1:])
        y1 = np.inner(weight,x)        
        y1 = y1 + bias    
        if y1 < 0:
            y1 = -1
        else:
            y1 = +1
        if y1 == y:            
            correctcount += 1
    accuracy = 100 * (float(correctcount)/float(total))
    return accuracy            
    
def crossvalidation(k, hyperparamEta, featuresize, epoch, hyperparamMu):

    file1 = open('Dataset/CVSplits/training00.data')
    file2 = open('Dataset/CVSplits/training01.data')
    file3 = open('Dataset/CVSplits/training02.data')
    file4 = open('Dataset/CVSplits/training03.data')
    file5 = open('Dataset/CVSplits/training04.data')
            
    dataset1 = file1.read()        
    data1 = libsvm2matrix(dataset1, featuresize)            
    dataset2 = file2.read()        
    data2 = libsvm2matrix(dataset2, featuresize)        
    dataset3 = file3.read()        
    data3 = libsvm2matrix(dataset3, featuresize)    
    dataset4 = file4.read()
    data4 = libsvm2matrix(dataset4, featuresize)    
    dataset5 = file5.read()
    data5 = libsvm2matrix(dataset5, featuresize)    

    maxaccuracy = 0
    out = []
    for mu in hyperparamMu:
        print "Cross Validation for mu = ",mu
        for eta in hyperparamEta:
            #print "Cross Validation for eta = ",eta
            valaccuracy = []
            for i3 in range(k):            
                if i3 == 0:
                    train = data2 + data3 + data4 + data5
                    test = dataset1
                    #text1 = 'training on Combining training01.data,training02.data,training03.data,training04.data and testing on training00.data'
                if i3 == 1:
                    train = data1 + data3 + data4 + data5
                    test = dataset2
                    #text1 = 'training on Combining training00.data,training02.data,training03.data,training04.data and testing on training01.data'
                if i3 == 2:
                    train = data1 + data2 + data4 + data5
                    test = dataset3
                    #text1 = 'training on Combining training00.data,training01.data,training03.data,training04.data and testing on training02.data'
                if i3 == 3:
                    train = data1 + data2 + data3 + data5
                    test = dataset4
                    #text1 = 'training on Combining training00.data,training01.data,training02.data,training04.data and testing on training03.data'
                if i3 == 4:
                    train = data1 + data2 + data3 + data4
                    test = dataset5
                    #text1 = 'training on Combining training00.data,training01.data,training02.data,training03.data and testing on training04.data'
                
                initialweight = [0.001]* featuresize
                bias = 0.001
                mxepacc = 0
                for ep in range(epoch):
                    np.random.shuffle(train)
                    neweta = float(eta)/float(eta+ep)
                    weight = perceptron(train,initialweight, bias, neweta, mu)
                    initialweight = weight[1:]                
                    acc = accuracyCalc(test, weight, featuresize)
                    if acc >= mxepacc:
                        mxepacc = acc
                        optwt = weight
                acc = accuracyCalc(test, optwt, featuresize)
                #print text1
                #print "Accuracy: ", acc
                valaccuracy.append(acc)
            avgacc = float(sum(valaccuracy))/float(len(valaccuracy))
            accuracystd = np.std(valaccuracy)
            print "For eta : {}, Average Accuracy = {}".format(eta, avgacc)            
            if avgacc >= maxaccuracy:
                maxaccuracy = avgacc
                out = []
                out.append(eta)
                out.append(mu)
        print '\n'
    print "(a) The best Hyperparameter: Eta = {}, Mu = {}".format(out[0],out[-1])
    print "(b) The cross validation accuracy for best hyperparameter: ", maxaccuracy
    return out            

#Driving Code

#Reading Inputs for training, Dev and Test Data            
ftrain = open('Dataset/phishing.train')
dataset = ftrain.read()

fdev = open('Dataset/phishing.dev')
devdata = fdev.read()

ftest = open('Dataset/phishing.test')
testdata = ftest.read()

#Initialising to drive the Code
featuresize = getFeatureDim(dataset)
t = libsvm2matrix(dataset,featuresize)
initialweight = [0.001]* featuresize
bias = 0.001
hyperparamEta = [1, 0.1,0.01]
hyperparamMu = [1, 0.1,0.01]
epoch = 10
etamu = crossvalidation(5, hyperparamEta, featuresize, epoch, hyperparamMu)
eta = etamu[0]
mu = etamu[-1]
#print "Best eta", eta
#print "Best mu", mu


#Checking on Dev data
mxdevaccr = 0
devaccuracy = []
updt = []
for ep in range(20):    
    np.random.shuffle(t)
    neweta = float(eta)/float(1+ep)
    weight = perceptron(t,initialweight, bias, eta, mu)
    u = updatecount(t,initialweight, bias, eta, mu)
    updt.append(u)
    initialweight = weight[1:]    
    accuracy = accuracyCalc(devdata, weight, featuresize)
    devaccuracy.append(accuracy)
    if accuracy > mxdevaccr:
        mxdevaccr = accuracy
        optweight = weight
        epo = ep+1
updatesum = sum(updt[:epo])
avgupdate = round(float(sum(updt[:epo]))/float(epo),3)        
print "(c) Total updates on training dataset from Epoch #1 to Best Epoch({}) is {}".format(epo,updatesum)
print "     And average update on every epoch till the best epoch is {}".format(avgupdate)
print "(d) Best Dev Set accuracy:{} (for Epoch #{})".format(mxdevaccr, epo)
print "     Dev Set accuracy for last Epoch {}".format(accuracy)

#Test on Test data using optimal weight from Dev set
accuracy = accuracyCalc(testdata, optweight, featuresize)
print "(e) Test Accuracy:", accuracy


'''
#Plot: Accuracy Vs Epochs
plt.plot(range(1,21),devaccuracy)
plt.ylabel('Development Set Accuracy')
plt.xlabel('Epochs')
plt.show()
'''
