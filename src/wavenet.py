# These are all the modules we'll be using later.

from __future__ import division
import numpy as np
import tensorflow as tf
import scipy.io   
import h5py
import matplotlib.pyplot as plt
import os
import sys
import random
import scipy.io.wavfile



#-------------------------------------Add working directory to path-----------------------------------------------

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.insert(0,'/home/student/s/snambusubram/Documents/Edge-Computing')


# Save the variables in a log/directory during training

save_path = "/home/student/s/snambusubram/Documents/Edge-Computing/wavenet_logs"
if not os.path.exists(save_path):
    os.makedirs(save_path)


# -------------------------Get some insights and information about the training data-----------------------------


# Location of the wav file in the file system.
fileName1 = '/home/student/s/snambusubram/Documents/Edge-Computing/dataset/UMAPiano-DB-Poly-1/UMAPiano-DB-A0-NO-F.wav'
fileName2 = '/home/student/s/snambusubram/Documents/Edge-Computing/dataset/UMAPiano-DB-Poly-1/UMAPiano-DB-A0-NO-M.wav'
# Loads sample rate (bps) and signal data (wav). Notice that Python
sample_rate1, data1 = scipy.io.wavfile.read(fileName1)
sample_rate2, data2 = scipy.io.wavfile.read(fileName2)


# Print in sdout the sample rate, number of items and duration in seconds of the wav file
print("Sample rate1 %s  data size1 %s  duration1: %s seconds"%(sample_rate1,data1.shape,len(data1)/sample_rate1))
print("Sample rate2 %s  data size2 %s  duration2: %s seconds"%(sample_rate2,data1.shape,len(data2)/sample_rate2))

# Plot the wave file and get insight about the sample. Here we test first 100 samples of the wav file

# fileName_arr = np.fromfile(open(fileName),np.int16)[0:96270]

# plt.plot(fileName_arr)
plt.plot(data1)
plt.plot(data2)
plt.show()

#---------------------------------------------------------------------------------------------------------------

# Split the data into train,validation and test 

dataset_path = '/home/student/s/snambusubram/Documents/Edge-Computing/dataset/UMAPiano-DB-Poly-1'	
dir_list_len = len(os.listdir(dataset_path))
print("Number of files in the Dataset ",dir_list_len)
training_data_arr = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

# Add the FileData instances to a list or another class and dump that to a file







