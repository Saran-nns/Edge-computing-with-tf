from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import scipy.io   
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile

def create_dir(path:str):
    """Create directory if the path doest exist

    Args:
        path ([type]): [description]
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(' {} Path already exists'.format(path))
    return None
        
def read_wav(file:str):
    """Read the wave file and return the sample rate and the signal as numpy array

    Args:
        file (wav): File/ absolute path of the file

    Returns:
        sample rate (int): Sample rate of the signal
        data (array): Signal
    """
    sample_rate, data = scipy.io.wavfile.read(file)
    return sample_rate, data

def describe(path:str):
    """Describe the audio file

    Args:
        path (str): Absolute path of the file

    Returns:
        Print sample rate, data size and duration of the signal
    """
    sample_rate, data = read_wav(path)
    print("Sample rate {} | Data size {} | Duration: {} seconds ".format(sample_rate,data.shape,len(data)/sample_rate))
    return None

def set_sample_rate(dataPath: str, fileName: str, sample_rate: int, writeFolderPath: str):
    """Change the sample rate of the audio signal

    Args:
        dataPath (str): Foldr path 
        fileName (str): File name
        sample_rate (int): Target sample rate
        writeFolderPath (str): Target folder path
    """
    _,data = scipy.io.wavfile.read(dataPath)
    scipy.io.wavfile.write('%s/%i.wav'%(writeFolderPath,fileName), sample_rate, data) 
    
    return None



    
    