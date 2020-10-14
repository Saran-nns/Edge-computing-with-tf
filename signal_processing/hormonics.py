import parabolic
from pylab import subplot, plot, log, copy, show

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
    
        nyq = 0.5 * fs    # Nyquist frequency
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