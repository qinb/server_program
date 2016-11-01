#################################################  
#SVM: support vector machine  
#Author : qinb  
#Email  : 
###############################################  

from numpy import * 
import numpy as np
import time  
import matplotlib.pyplot as plt   


seq_length = 1
input_size = 2
toler = 0.001
C = 0.6
maxIter = 60
Inf = 10000


def load_data(dataset):
    print('loading data ...')
    data = open(dataset, 'r')
    prices = np.zeros((Inf,3))
    num = 0
    for line in data:
        prices[num] = line.split('\n')[0].split('\t')
        num += 1
	#prices = clean_data(prices[1::])
    #return prices[1::]
    return prices[0:num]


# the 2D data
def gen_sample(data):

    samples = np.zeros([col,seq_length*input_size])

    for i in range(len(samples)):
		#calculate the limit up as the label	
        cur_high = float(data[i+ seq_length][1])
        pre_close = float(data[i+ seq_length-1][2])
        cur_limit_up = round(pre_close*1.08,2)

		# class 1 means that the day happens to limit up
		#otherwise,it doesn't
        if (cur_high >= cur_limit_up):
            samples[i][col] = 1 
        else:
            samples[i][col] = 0

        for j in range(seq_length):
            for k in range(input_size):
                samples[i][j*input_size+k] = data[i+j][k]
    return samples



#calulate kernel value  
def calcKernelValue(matrix_x, sample_x, kernelOption):  
  kernelType = kernelOption[0]  
  numSamples = matrix_x.shape[0]  
  kernelValue = np.zeros(numSamples)
  if kernelType == 'linear':  
      kernelValue = matrix_x * sample_x.T  
  elif kernelType == 'rbf':  
      sigma = kernelOption[1]  
      if sigma == 0:  
          sigma = 1.0  
      for i in xrange(numSamples): 
          sumtmp  = 0  
          diff = matrix_x[i,:] - sample_x  
          for k in range(seq_length*input_size):
              sumtmp += diff[k]*diff[k] 
          kernelValue[i] = exp(float(sumtmp) / (-2.0 * sigma**2))  
  else:  
      raise NameError('Not support kernel type! You can use linear or rbf!')  
  return kernelValue  


#calculate kernel matrix given train set and kernel type 
#calculate all the sample's multiply storing in the kernelMatrix
def calcKernelMatrix(train_x, kernelOption):  
  numSamples = train_x.shape[0]  
  kernelMatrix = np.zeros((numSamples, numSamples))  
  for i in xrange(numSamples):  
      kernelMatrix[i] = calcKernelValue(train_x, train_x[i, :], kernelOption)  
  return kernelMatrix  


#define a struct just for storing variables and data  
class SVMStruct:  
  def __init__(self, dataSet, labels, C, toler, kernelOption):  
      self.train_x = dataSet # each row stands for a sample  
      self.train_y = labels  # corresponding label  
      self.C = C             # slack variable  
      self.toler = toler     # termination condition for iteration  
      self.numSamples = len(dataSet) # number of samples  
      self.alphas = np.zeros((self.numSamples, 1)) # Lagrange factors for all samples  
      self.b = 0  
      self.errorCache = np.zeros((self.numSamples, 2))
      self.kernelOpt = kernelOption  
      self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)  

        
#calculate the error for alpha k  
# alpha_k is the array index
def calcError(svm, alpha_k):  
  #output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b) 
  
  output_k = np.zeros(svm.numSamples)
  for i in range(svm.numSamples):
      output_k[i] = svm.alphas[i]*svm.train_y[i]

  outputtmp = 0
  for i in range(svm.numSamples):
      outputtmp += output_k[i]*svm.kernelMat[i,alpha_k] 

  error_k = outputtmp - svm.train_y[alpha_k]  
  return error_k  


#update the error cache for alpha k after optimize alpha k  
def updateError(svm, alpha_k):  
  error = calcError(svm, alpha_k)  
  svm.errorCache[alpha_k] = [1, error]  


#select alpha j which has the biggest step  
def selectAlpha_j(svm, alpha_i, error_i):  
  svm.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)  
  candidateAlphaList = nonzero(svm.errorCache[:, 0]) # mat.A return array  
  maxStep = 0; alpha_j = 0; error_j = 0  

  # find the alpha with max iterative step  
  if len(candidateAlphaList) > 1:  
      for alpha_k in candidateAlphaList:  
          if alpha_k == alpha_i:   
              continue  
          error_k = calcError(svm, alpha_k)  
          if abs(error_k - error_i) > maxStep:  
              maxStep = abs(error_k - error_i)  
              alpha_j = alpha_k  
              error_j = error_k  
  # if came in this loop first time, we select alpha j randomly  
  else:             
      alpha_j = alpha_i  
      while alpha_j == alpha_i:  
          alpha_j = int(random.uniform(0, svm.numSamples))  
      error_j = calcError(svm, alpha_j)  
    
  return alpha_j, error_j  


#the inner loop for optimizing alpha i and alpha j  
def innerLoop(svm, alpha_i):  
    error_i = calcError(svm, alpha_i)  
    #print(error_i) 
   ### check and pick up the alpha who violates the KKT condition  
   ## satisfy KKT condition  
   # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)  
   # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)  
   # 3) yi*f(i) <= 1 and alpha == C (between the boundary)  
   ## violate KKT condition  
   # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so  
   # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)   
   # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)  
   # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized  
    if (svm.train_y[alpha_i]*error_i < -svm.toler)and(svm.alphas[alpha_i] < svm.C)or(svm.train_y[alpha_i]*error_i > svm.toler)and(svm.alphas[alpha_i] > 0):
	#or(svm.train_y[alpha_i]*error_i==svm.toler)and(svm.alphas[alpha_i] == 0 or svm.alphas[alpha_i] == svm.C):  
       # step 1: select alpha j  
       alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)  
       alpha_i_old = svm.alphas[alpha_i].copy()  
       alpha_j_old = svm.alphas[alpha_j].copy()  
 
       # step 2: calculate the boundary L and H for alpha j  
       if svm.train_y[alpha_i] != svm.train_y[alpha_j]:  
           L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])  
           H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])  
       else:  
           L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)  
           H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])  
       if L == H:  
           return 0  
 
       # step 3: calculate eta (the similarity of sample i and j)  
       eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i]- svm.kernelMat[alpha_j, alpha_j]  
       if eta >= 0:  
           return 0  
 
       # step 4: update alpha j  
       svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta  
 
       # step 5: clip alpha j  
       if svm.alphas[alpha_j] > H:  
           svm.alphas[alpha_j] = H  
       if svm.alphas[alpha_j] < L:  
           svm.alphas[alpha_j] = L  
 
       # step 6: if alpha j not moving enough, just return       
       if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:  
           updateError(svm, alpha_j)  
           return 0  
 
       # step 7: update alpha i after optimizing aipha j  
       svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j]*(alpha_j_old - svm.alphas[alpha_j])  
 
       # step 8: update threshold b 
       print('error_i = %f' %error_i)
       print('error_j = %f' %error_j)
       print('svm.b = %f' %svm.b)
       b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old)  * svm.kernelMat[alpha_i, alpha_i] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernelMat[alpha_i, alpha_j] * svm.kernelMat[alpha_i, alpha_j]  
       b2 = svm.b - error_j - svm.train_y[alpha_i] *(svm.alphas[alpha_i] - alpha_i_old) * svm.kernelMat[alpha_i, alpha_j] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernelMat[alpha_j, alpha_j]  
       if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):  
           svm.b = b1  
       elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):  
           svm.b = b2  
       else:  
           svm.b = (b1 + b2) / 2.0  
 
       # step 9: update error cache for alpha i, j after optimize alpha i, j and b  
       updateError(svm, alpha_j)  
       updateError(svm, alpha_i)  
       return 1  
    else:  
       return 0  
 
 
#the main training procedure  
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 1.0)):  
   # calculate training time  
   startTime = time.time()  
 
   # init data struct for svm  
   svm = SVMStruct(train_x, train_y, C, toler, kernelOption)  
     
   # start training  
   entireSet = True  
   alphaPairsChanged = 0  
   iterCount = 0  
   # Iteration termination condition:  
   #   Condition 1: reach max iteration  
   #   Condition 2: no alpha changed after going through all samples,  
   #                in other words, all alpha (samples) fit KKT condition  
   while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):  
       alphaPairsChanged = 0  
 
       # update alphas over all training examples  
       if entireSet:  
           for i in xrange(svm.numSamples):  
               alphaPairsChanged += innerLoop(svm, i)  
           print '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  
           iterCount += 1  
       # update alphas over examples where alpha is not 0 & not C (not on boundary)  
       else:  
           nonBoundAlphasList = nonzero((svm.alphas > 0) * (svm.alphas < svm.C))[0] 
           #print(nonBoundAlphasList)
           for i in nonBoundAlphasList: 
               #print('i = %d' %i)
               alphaPairsChanged += innerLoop(svm, i)  
           print '---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  
           iterCount += 1  
 
       # alternate loop over all examples and non-boundary examples  
       if entireSet:  
           entireSet = False  
       elif alphaPairsChanged == 0:  
           entireSet = True  
 
   print 'training complete! Took %fs!' % (time.time() - startTime)  
   return svm  
 
 
# testing your trained svm model given test set  
def testSVM(svm, test_x, test_y):  
   test_x = test_x 
   test_y = test_y  
   numTestSamples = test_x.shape[0] 
   supportVectorsIndex = nonzero(svm.alphas > 0)[0]
   print(supportVectorsIndex)
   supportVectors      = svm.train_x[supportVectorsIndex]  
   supportVectorLabels = svm.train_y[supportVectorsIndex]  
   supportVectorAlphas = svm.alphas[supportVectorsIndex]  
   matchCount = 0  
   for i in xrange(numTestSamples):  
       kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt) 
       sumtmp = np.zeros(len(supportVectorsIndex))
       tmp = 0
       for j in range(len(supportVectorsIndex)):
          sumtmp[j] = supportVectorLabels[j]*supportVectorAlphas[j]
       for k in range(len(supportVectorsIndex)):
          tmp += kernelValue[k]*sumtmp[k]*kernelValue[k] 
       predict = tmp + svm.b
       print(svm.b)
       if sign(predict) == sign(test_y[i]):  
           matchCount += 1  
   accuracy = float(matchCount) / numTestSamples  
   return accuracy  
 
 
# show your trained svm model only available with 2-D data  
def showSVM(svm):  
   if svm.train_x.shape[1] != 2:  
       print "Sorry! I can not draw because the dimension of your data is not 2!"  
       return 1  
 
   # draw all samples  
   for i in xrange(svm.numSamples):  
       if svm.train_y[i] == -1:  
           plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')  
       elif svm.train_y[i] == 1:  
           plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')  
 
   # mark support vectors  
   supportVectorsIndex = nonzero(svm.alphas > 0)[0] 
   #print(supportVectorsIndex)
   for i in supportVectorsIndex:  
       plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')  
     
   # draw the classify line  
   w = zeros((2, 1))  
   for i in supportVectorsIndex:  
       w[0] += svm.alphas[i] * svm.train_y[i]*svm.train_x[i, 0] 
       w[1] += svm.alphas[i] * svm.train_y[i]*svm.train_x[i, 1] 
   min_x = min(svm.train_x[:, 0]) 
   max_x = max(svm.train_x[:, 0])
   y_min_x = float(-svm.b - w[0] * min_x) / w[1]  
   y_max_x = float(-svm.b - w[0] * max_x) / w[1]  
   plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
   plt.show()  

if __name__ == '__main__':		
    train_data  = load_data('trainSet')
    test_data = load_data('testSet')
    #all_samples = gen_sample(train_data)
    all_samples = train_data
    train_samples = all_samples[:,0:(seq_length*input_size)]
    train_label = all_samples[:,seq_length*input_size]
    print('train model ......')
    svm = trainSVM(train_samples,train_label,C,toler,maxIter)	
	
    all_samples = test_data
    test_samples = all_samples[:,0:(seq_length*input_size)]
    test_label = all_samples[:,seq_length*input_size]
    print(testSVM(svm,test_samples,test_label))
    showSVM(svm)
