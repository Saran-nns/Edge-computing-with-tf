from numpy.core.records import array
import parabolic
from scipy.signal import butter, lfilter
from pylab import subplot, plot, log, copy, show
import numpy as np
import scipy
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import get_window
from math import ceil
from pylab import figure, imshow, clf, gray, xlabel, ylabel
import matplotlib.pyplot as plt
class HarmonicPowerSpectrum(object):
    
    def __init__(self,sig,f0,fs,order,maxharms):
        self.sig = sig
        self.f0 = f0
        self.fs = fs
        self.order = order
        self.maxharms = maxharms

    @property    
    def butter_bandpass(self):
        
        """Give the Sampling freq(fs),Bandpass window(f0) of filter, build the bandpass filter"""
    
        nyq = 0.5 * self.fs    # Nyquist frequency
        low = self.f0[0] / nyq
        high = self.f0[1] / nyq
        b, a = butter(self.order, [low, high], btype='band')  # Numerator (b) and denominator (a) polynomials of the IIR filter

        return b, a

    @property
    def butter_bandpass_filter(self):
    
        """ Apply bandpass filter to the given signal"""
    
        b, a = self.butter_bandpass
        y = lfilter(b, a, self.sig)    # Apply the filter to the signal

        return y 

    @property
    def hps(self):
    
        """Estimate peak frequency using harmonic product spectrum (HPS)"""
        
        y = self.butter_bandpass_filter
        window = y * scipy.signal.blackmanharris(len(y))  #Create window to search harmonics in signal slices

        # Harmonic product spectrum: Measures the maximum coincidence for harmonics for each spectral frame
        
        c = abs(np.fft.rfft(window))  # Compute the one-dimensional discrete Fourier Transform for real input.
        z = np.log(c)  # Fundamental frequency or pitch of the given signal  

        return z
    
def butter_bandpass(f0: float, fs: float, order: int):
    """Give the Sampling freq(fs),Bandpass window(f0) of filter, build the bandpass filter

    Args:
        f0 (float): Fundamental freq of the signal
        fs (float): sampling rate of the signal
        order (int): Order of the filter

    Returns:
        [type]: Numerators and denominators of the IIR filters
    """
    nyq = 0.5 * fs
    low = f0[0] / nyq
    high = f0[1] / nyq
    b, a = butter(order, [low, high], btype='band')  # Numerator (b) and denominator (a) polynomials of the IIR filter
    
    return b, a

def butter_bandpass_filter(sig: array, f0: float, fs: float, order:int):
    """Apply bandpass filter (IIR) to the given signal

    Args:
        sig (array): Signal
        f0 (float): Fundamental freq of the signal
        fs (float): Sampling rate
        order (int): Order of the filter

    Returns:
        array: IIR filter applied signal
    """
    b, a = butter_bandpass(f0, fs,order)
    y = lfilter(b, a, sig)    # Apply the filter to the signal
    
    return y  

def hps(sig: array,fs: float,maxharms:int):
    """
    Estimate peak frequency using harmonic product spectrum (HPS)
    Harmonic product spectrum: Measures the maximum coincidence for harmonics for each spectral frame

    Args:
        sig (array): Signal
        fs (float): [description]
        maxharms (int): [description]
    """
    window = sig * scipy.signal.blackmanharris(len(sig))
    c = abs(np.fft.rfft(window))  # Compute the one-dimensional discrete Fourier Transform for real input.
    pitch = np.log(c)
    return c, pitch

def plot_dft(sig: array,fs: float,maxharms:int):
    """
    Estimate peak frequency using harmonic product spectrum (HPS)
    Harmonic product spectrum: Measures the maximum coincidence for harmonics for each spectral frame

    Args:
        sig (array): Signal
        fs (float): [description]
        maxharms (int): [description]
    returns plots
    """
    window = sig * scipy.signal.blackmanharris(len(sig))
    c = abs(np.fft.rfft(window))  # Compute the one-dimensional discrete Fourier Transform for real input.
    pitch = np.log(c)
    plt.plot(c)
    plt.title("Discrete fourier transform of signal")
    plt.plot(pitch)
    plt.title("Max Harmonics for the range same as fundamental frequencies")
    return plt.show()

def get_max_harmonics(sig: array,fs: float,maxharms:int):
    """
    Estimate peak frequency using harmonic product spectrum (HPS)
    Harmonic product spectrum: Measures the maximum coincidence for harmonics for each spectral frame
    Search for a maximum value of a range of possible fundamental frequencies
    Args:
        sig (array): Signal
        fs (float): [description]
        maxharms (int): [description]
    """  
    c, pitch = hps(sig,fs,maxharms)
    for x in range(2, maxharms):
        a = copy(c[::x])  # Should average or maximum instead of decimating
        c = c[:len(a)]
        i = np.argmax(abs(c))
        c *= a
        plt.title("Max Harmonics for the range of %d times the fundamental frequencies"%x)
        plt.plot(maxharms, x)
        plt.plot(np.log(c))
    show()
