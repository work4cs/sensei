import pickle
import pandas as pd
import numpy as np
import string
import re
from itertools import *
import itertools
class EVA(object):
  def __init__(self,test=None,dTest=None,savePath=None,nameNext='nextP',nameData='nameNon',labelSeg='labelSeg',direction='Forward',dEval=None):
    self.test = test
    self.dTest = dTest
    self.dEval = dEval
    self.nameNext = nameNext
    if not (dTest is None):
        self.dTest['next'] = self.dTest[nameNext].copy()
    self.nameData = nameData 
    self.labelSeg = labelSeg 
    self.direction = direction#Forward Backward Two Bi
    self.savePath = savePath 

    self.diff = None

    if self.direction =='Backward': #dTest.p keeps back
      self.dTest = self.dTest[::-1]
      
    if dTest is not None and test is not None:
      self.addTruth()
      self.diff = self.test['diff'].values.tolist()
  
  def addTruth(self):
    assert ''.join(self.test[self.nameData].values.tolist()) == ''.join(self.dTest['char'].values.tolist())
    
    truth = ''.join(self.test['labelOri'].values.tolist())
    assert len(truth) == len(self.dTest)
    truth = truth.translate(str.maketrans('bio','010'))
    
    if self.direction == 'Backward':
      seg = [i.span() for i in re.finditer('1*0', truth)]
    else:
      seg = [i.span() for i in re.finditer('01*', truth)]
    truth = list(truth)
    for i in range(len(seg)):
      #order of assign: single char in a seg should be 0
      truth[seg[i][0]] = '1'
      truth[seg[i][1]-1] = '0'
    truth = list(map(int, truth))
    self.dTest['truth'] = list(truth)
    # return truth
    
  def getDiff(self,labelSeg):
    diffList = []

    for index ,s in enumerate(labelSeg):
      diff = []  
      for i in range(len(s)-1):
        if s[i][1] != s[i+1][0]:
          diff.append((s[i][1],s[i+1][0]))
    
      diffList.append(diff)

    return diffList

  def split_by_lengths(self, seq, num):
    it = iter(seq)
    out =  [x for x in (list(islice(it, n)) for n in num) if x]
    #remain = list(it)
    return out #if not remain else out + [remain]

  def getEval(self, test=None, dTest=None):
    if test is None:
      test = self.test
      
    if dTest is None:
      dTest = self.dTest
      
    length = test[self.nameData].apply(lambda x: len(x)).values.tolist()
  
    assert self.split_by_lengths(dTest['char'].values.tolist(), length) == [list(x) for x in test[self.nameData].values.tolist()]
    nextP = self.split_by_lengths(dTest['next'].values.tolist(), length)
    dEval = {'nameOri':test[self.nameData].values.tolist(),'GT':test[self.nameData].values.tolist(),'show':test[self.nameData].values.tolist(),'next':nextP, 'labelSeg':test[self.labelSeg].values.tolist()}
    self.dEval = pd.DataFrame(dEval)
    return self.dEval

  #nextP prob prediction and should be transfered to seg prediction
  # ignore=0: not ignore; ignore=1: only ignore seg o; ignore=2: ignore seg and side o 
  def getPred(self, nextP, threshold=0.5, ignore=2, diff=None):
    pred = []
    for i,p in enumerate(nextP):
      s = [str(int(x<threshold)) for x in p]
      if self.direction == 'Backward':
        s[0] = '1' ##
        regex = '10*'
      else: 
        s[-1] = '1' ##
        regex = '0*1'
      s = ''.join(s)
      
      segs = [i.span() for i in re.finditer(regex, s)]
      
      pred.append(segs)
    
    predIgnore = self.ignoreO(pred, ignore=ignore, diff=diff)
    return pred, predIgnore
  
  #diff is a list,  output a list
  def ignoreO(self, pred, ignore=2, model='chunk', diff=None):
    
    if (diff == None) and (self.diff != None):
      diff = self.diff

    newPred = []
    if model == 'chunk':
      for i,segs in enumerate(pred):
        newSegs = []
  
        for j,seg in enumerate(segs):
          add = True
          for d in range(len(diff[i])):
            if seg[0] >= diff[i][d][0] and seg[1] <= diff[i][d][1]:
              add = False
              break
            if ignore >= 2:
              if seg[0] >= diff[i][d][0] and seg[0] < diff[i][d][1]:
                seg = (diff[i][d][1], seg[1])
    
              if seg[1] > diff[i][d][0] and seg[1] <= diff[i][d][1]:
                seg = (seg[0], diff[i][d][0])
          if add:
            newSegs.append(seg)
              
        newPred.append(newSegs)
    elif model == 'tag':
      for i,segs in enumerate(pred):
        newSegs = []
  
        for j,seg in enumerate(segs):
          add = True
          if ignore == 1 and seg[2] == '':
              add = False
              
          if add:
            newSegs.append(seg)
              
        newPred.append(newSegs)  
    return newPred
  
  # macro avg F1: evaluate individual and then avg    
  def evaluate(self, y, pred):
    precision = 0
    recall = 0
    F1 = 0
    index = 0
    for i,j in zip(y,pred):
      tp = len(set(i).intersection(j))
      if len(j) != 0:
        p = tp/len(j)
      else:
        p = 0
      precision += p

      if len(i) != 0:
        r = tp/len(i)
      else:
        r = 0
        #print(index,i,j)
      recall += r
      
      if (p+r) == 0:
        f = 0
      else:
        f = 2*p*r/(p+r)
      F1+=f
      
      index += 1

    return F1/len(y), precision/len(y), recall/len(y)
  
  def evaluate2(self, y, pred):
    precision = 0
    recall = 0
    F1 = 0
    tp = 0
    p = 0
    r = 0
    
    for i,j in zip(y,pred):
      tp += len(set(i).intersection(j))
      p += len(j)
      r += len(i)

    precision = tp/(p+1e-9)
    recall = tp/(r+1e-9)
    F1=2*precision*recall/(precision+recall+1e-9)
      
    return F1, precision, recall
  
  #verbose=2 store and report, verbose=1 report, verbose=0 nothing
  def reportByThresholds(self, thresholds=[0.5], split=0, verbose=0, ignore=2, dEval=None):
    if dEval == None:
      dEval = self.getEval()
    
    split = int(len(dEval)*split)
    y = dEval['labelSeg'].values.tolist()
    nextP = dEval['next'].values.tolist()
    assert len(y)==len(nextP)
    
    if split == 0:
      resultsList = []
      F1List = [] #could only store resultsList
      for t in thresholds:
        ori,pred = self.getPred(nextP, threshold=t, ignore=ignore, diff=self.diff)
        dEval['predIgnore'] = pred #devPred + testPred
        dEval['pred'] = ori #devOri + testOri
        if ignore > 0:
          results = self.evaluate(y, pred)
        else:
          results = self.evaluate(y, ori)
        resultsList.append(results)
        F1List.append(results[0])
      
      if verbose == 1:
        print('dataset size: y={}, nextP={}'.format(len(y), len(nextP)))
      
      index = np.argmax(F1List)
      F1 = F1List[index]
      assert F1 == np.max(F1List)
      bestThreshold = thresholds[index]
      
      if len(thresholds) > 1 and verbose==2:
        dStats = {}
        dStats['bestThreshold'] = bestThreshold
        dStats['F1List'] = F1List
        pickle.dump(dStats, open('./data/'+self.savePath+'GS.p', "wb"  ))
      return resultsList[index], bestThreshold
    
    ##with dev, split != 0
    devY = y[:split]
    devNextP = nextP[:split]
    testY = y[split:]
    testNextP = nextP[split:]
    assert len(devY)==len(devNextP)
    assert len(testY)==len(testNextP)
      
    F1List = []
    # devPredList = []
    # devOriList = []
    for t in thresholds:
      devOri, devPred = self.getPred(devNextP, threshold=t, ignore=ignore, diff=self.diff[:split])
      if ignore > 0:
        results = self.evaluate(devY, devPred)
      else:
        results = self.evaluate(devY, devOri)
      F1List.append(results[0])
      # devPredList.append(devPred)
      # devOriList.append(devOri)
    index = np.argmax(F1List)
    devF1 = F1List[index]

    assert devF1 == np.max(F1List)
    # devPred = devPredList[index]
    # devOri = devOriList[index]
    bestThreshold = thresholds[index]

    testOri,testPred = self.getPred(testNextP, threshold=bestThreshold, ignore=ignore, diff=self.diff[split:])

    ori,pred = self.getPred(nextP, threshold=bestThreshold, ignore=ignore)
    # F1, precision, recall = self.evaluate(y, allPred)
    dEval['predIgnore'] = pred #devPred + testPred
    dEval['pred'] = ori #devOri + testOri
    
    self.dEval = dEval
    
    if verbose != 0:
      print('dataset size: devY={}, devNextP={}, testY={}, testNextP={}'.format( len(devY), len(devNextP), len(testY), len(testNextP)))
    
    if ignore <= 0:
      pred = ori
      testPred = testOri
    
    if split == 0:
      return self.evaluate(y, pred), bestThreshold
    else:
      return devF1, self.evaluate(testY, testPred), self.evaluate(y, pred), bestThreshold
  
  #visualization of segmentation
  def showSeg(self, predType='pred', dEval=None):
    if dEval is None:
      dEval = self.dEval.copy()
    else:
      dEval = dEval.copy()
    for index, row in dEval.iterrows(): 
      if len(row[predType]) == 0:
        return None

      idx = np.array(row[predType])
      j = row['nameOri']
      pos = 0
      for b,e in idx:
        b+= pos
        j = j[:b] + '|' + j[b:]
        pos += 1
        e+= pos
        j = j[:e] + '|' + j[e:]
        pos += 1
      j=j.replace('||','|')
      try:
        dEval['show'][index] = j
      except:
        a=0 #print(index)
    
      idx = np.array(row['labelSeg'])
      j = row['nameOri']
      pos = 0
      for b,e in idx:
        b+= pos
        j = j[:b] + '|' + j[b:]
        pos += 1
        e+= pos
        j = j[:e] + '|' + j[e:]
        pos += 1
      j=j.replace('||','|')
      dEval['GT'][index] = j
    return dEval

