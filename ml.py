# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 12:35:54 2018

@author: zz
"""

import sklearn
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#import sklearn
#a=np.load("available_data-130-130-26.npy")
#print a.shape
import pydicom
#import numpy as np
import os
#import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
data_dir='D:\zz\Desktop\cs168 data\data CS168'
patients=os.listdir(data_dir)[1:]
#print patients
workbook=xlrd.open_workbook('D:\zz\Desktop\cs168 data\labels.xlsx')
img_size=130
numslice=26
#print labelbook.sheet_names()
labelbook=workbook.sheet_by_index(0)

num_available_data=0
num_class0=0
num_class1=0
num_class2=0
num_class3=0
num_class4=0
num_class5=0
for num,patient in enumerate(patients):
    
    path=data_dir+'/'+patient+'/'+'DWI'
    if os.path.exists(path):

        if labelbook.row_values(int(patient)-1)[1]!='':
            num_available_data=num_available_data+1
            if labelbook.row_values(int(patient)-1)[1]=='A':
                num_class0=num_class0+1
            if labelbook.row_values(int(patient)-1)[1]=='B':
                num_class1=num_class1+1
            if labelbook.row_values(int(patient)-1)[1]=='C':
                num_class2=num_class2+1
            if labelbook.row_values(int(patient)-1)[1]=='D':
                num_class3=num_class3+1 
            if labelbook.row_values(int(patient)-1)[1]=='E':
                num_class4=num_class4+1
            if labelbook.row_values(int(patient)-1)[1]=='F':
                num_class5=num_class5+1
                
num_data_used=num_available_data
       

 
def preprocess_data(patient,labelbook,visualize=False):
     slices_name=[]
     label=labelbook.row_values(int(patient)-1)[1]
     original_label=label
     label1=np.array([])
     #unicodedata.normalize("NFKD",label).encode('ascii', 'ignore')
     path=data_dir+'/'+patient+'/'+'DWI'
    #print os.listdir(path)
     for s in os.listdir(path):
        if s[-3:]=='dcm':
            slices_name.append(s)
            #print slices_name
     slices=[pydicom.read_file(path+'/'+s) for s in slices_name]
     slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
     slices_data=np.array([oneslice.pixel_array for oneslice in slices])
     standard_shape=[numslice,img_size,img_size]
     current_shape=[len(slices),slices[0].pixel_array.shape[0],slices[0].pixel_array.shape[1]]
     shape_factor=[w/float(f) for w,f in zip(standard_shape,current_shape)]
     standard_data=zoom(slices_data,shape_factor)
     #print standard_data.shape
     #new_form_data=standard_data[13,:,:]
     #new_standard_data=new_form_data.flatten()
     new_standard_data=standard_data.reshape(img_size*img_size*numslice)
     
     
     
     
     
     
     #print label.shape
     #print label
     if label=='A':
         label1=0
         
     if label=='B':
         
         label1=0 
     if label=='C':
         label1=0
     if label=='D':
         label1=0   
     if label=='E':
         label1=1
     if label=='F':
         label1=0 
     #print label
         
     if visualize:
         fig=plt.figure()
         for num,each_slice in enumerate(slices):
             y=fig.add_subplot(4,5,num+1)
             y.imshow(each_slice,cmap='gray')
         #plt.imshow(slices[0].pixel_array)
         plt.show()
     return new_standard_data,label1,original_label



data=np.zeros((num_data_used,img_size*img_size*numslice))
labels=np.zeros(num_data_used)
original_labels=[]
i=0
for num,patient in enumerate(patients):
    
    path=data_dir+'/'+patient+'/'+'DWI'
    if os.path.exists(path):

        if labelbook.row_values(int(patient)-1)[1]!='':
           img_data,label1,original_l=preprocess_data(patient,labelbook)
           data[i,:]=img_data
           labels[i]=label1
           original_labels.append(original_l)
           i=i+1

#test class 'E'
#mycounter=0
class0_index_list=[]
class1_index_list=[]
class2_index_list=[]
class3_index_list=[]

class4_index_list=[]

class5_index_list=[]

#other_index_list=[]
train_index_list=[]
test_index_list=[]
#indices_test_class_we_want=[]
for j in range(len(labels)):
    
    if original_labels[j]=='A':
        class0_index_list.append(j)
    
    if original_labels[j]=='B':
        class1_index_list.append(j)
        
    if original_labels[j]=='C':
        class2_index_list.append(j)
    
    if original_labels[j]=='D':
        class3_index_list.append(j) 
        
    if original_labels[j]=='E':
        class4_index_list.append(j)
    if original_labels[j]=='F':
        class5_index_list.append(j)
        
        
        
        
indices_train=[]
indices_test=[]        
        
l1=class4_index_list[:19]
l2=class0_index_list[:40]
l3=class1_index_list[:26]
l4=class2_index_list[:44]
l5=class3_index_list[:4]

l6=class5_index_list[:48]
indices_train=l1+l2+l3+l4+l5+l6
for i in range(len(original_labels)):
    if i not in indices_train:
         indices_test.append(i)        
        
print len(indices_train)
print len(indices_test)         
    
training_scaled=preprocessing.scale(data)
x_train=training_scaled[indices_train,:]
y_train=labels[indices_train]
x_test=training_scaled[indices_test,:]
y_test=labels[indices_test]


#for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
#for kernel in ['rbf']:
	# use SVM with current kernel 



    #model=sklearn.kernel_ridge.KernelRidge(kernel=kernel,degree=3)

#    model = SVC(kernel=kernel,class_weight='balanced')
#    model.fit(x_train, y_train)

model=GradientBoostingClassifier(n_estimators=4600,learning_rate=1) 
#model = RandomForestClassifier(n_estimators=140,class_weight='balanced')
    #model.fit(x_train, y_train)
#

#	# evaluate the trained model on the test set
#testAccuracy = model.score(x_test, y_test)
#
#print("Final results of RandomForestClassifier: testing accuracy of %f%%"%(testAccuracy * 100))    
#model=GradientBoostingClassifier(n_estimators=2000,learning_rate=1)
model.fit(x_train,y_train) 
##print model.score(X_test,Y_test)  
#           
## evaluate the trained model on the test set
y_pred = model.predict(x_test)
#print y_pred
#print Y_test
#    # testAccuracy = model.score(X_test, Y_test)
testAccuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
roc_auc_score=sklearn.metrics.roc_auc_score(y_test,y_pred)
print testAccuracy
print roc_auc_score
#precision, recall, _ = sklearn.metrics.precision_recall_curve(Y_test, y_pred)
#    
#    
#print("accuracy", testAccuracy * 100)
#    # print(testAccuracy2 * 100)
#
#print("roc_auc", roc_auc_score)
#print("precision", precision)
#print("recall", recall)
#
#
#plt.step(recall, precision, color='b', alpha=0.2,
#         where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2,
#                 color='b')
#
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('2-class Precision-Recall curve')
#plt.show()           
