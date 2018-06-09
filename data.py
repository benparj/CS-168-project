# -*- coding: utf-8 -*-
"""
Created on Mon May 28 00:29:22 2018

@author: zz
"""

import pydicom
import numpy as np
import os
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
data_dir='data CS168'
patients=os.listdir(data_dir)[1:]
#print patients
workbook=xlrd.open_workbook('labels.xlsx')
img_size=130
numslice=26
#print labelbook.sheet_names()
labelbook=workbook.sheet_by_index(0)
#print labelbook.row_values(240)

#for patient in patients[:10]:
#    slices_name=[]
#    label=labelbook.row_values(int(patient)-1)[1]
#    if label=='A':
#        label=np.array([1,0,0,0,0,0])
#        print label
##    if label=='':
##        print "hello"
#        
#    #print label==''
#    path=data_dir+'/'+patient+'/'+'DWI'
#    #print path
#    #print os.listdir(path)
#    if os.path.exists(path):
#     for s in os.listdir(path):
#       if s[-3:]=='dcm':
#            slices_name.append(s)
#            #print slices_name
#     slices=[pydicom.read_file(path+'/'+s) for s in slices_name]
#     slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
#     slices_data=np.array([oneslice.pixel_array for oneslice in slices])
#     standard_shape=[numslice,img_size,img_size]
#     current_shape=[len(slices),slices[0].pixel_array.shape[0],slices[0].pixel_array.shape[1]]
#     shape_factor=[w/float(f) for w,f in zip(standard_shape,current_shape)]
#     standard_data=zoom(slices_data,shape_factor)
#     #print standard_data.shape
     
     
     
     
     #fig=plt.figure
#     for num,each_slice in enumerate(slices):
#        #y = fig.subplot(6,8,num+1)
#        new_img = cv2.resize(np.array(each_slice.pixel_array),(img_size,img_size))
#        
#        plt.imshow(new_img)
##    #plt.imshow(slices[0].pixel_array,cmap='gray')
#     #plt.imshow(slices[0].pixel_array)
#        plt.show()
#    a=np.array(slices[0].pixel_array)
#    #print a
#        plt.imshow(new_img)
#        plt.show()
     #print len(slices)
     #print slices[25]
#     if slices[0].PixelSpacing[0]==1.6923077106476:
#         n=n+1
#print n
 
def preprocess_data(patient,labelbook,visualize=False):
     slices_name=[]
     label=labelbook.row_values(int(patient)-1)[1]
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
     
     
     
     
     
     
     #print label.shape
     #print label
     if label=='A':
         label1=np.array([1,0,0,0,0,0])
     if label=='B':
         
         label1=np.array([0,1,0,0,0,0])    
     if label=='C':
         label1=np.array([0,1,0,0,0,0])
     if label=='D':
         label1=np.array([0,1,0,0,0,0])    
     if label=='E':
         label1=np.array([0,1,0,0,0,0])
     if label=='F':
         label1=np.array([0,1,0,0,0,0]) 
     #print label
         
     if visualize:
         fig=plt.figure()
         for num,each_slice in enumerate(slices):
             y=fig.add_subplot(4,5,num+1)
             y.imshow(each_slice,cmap='gray')
         #plt.imshow(slices[0].pixel_array)
         plt.show()
     return standard_data,label1

#patient = patients[0]
#img_data,label=preprocess_data(patient,labelbook)


available_data=[]
for num,patient in enumerate(patients):
    if num % 20==0:
        print(num)
    path=data_dir+'/'+patient+'/'+'DWI'
    if os.path.exists(path):

        if labelbook.row_values(int(patient)-1)[1]!='':
           img_data,label1=preprocess_data(patient,labelbook)
##        
           available_data.append([img_data,label1])
np.save('available_data-{}-{}-{}-class1.npy'.format(img_size,img_size,numslice),available_data)    