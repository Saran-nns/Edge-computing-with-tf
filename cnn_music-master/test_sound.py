import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
scaled = np.int16(data/np.max(np.abs(data)) * 32767)
write('test_sound.wav', 44100, scaled)

#----------------------------ANALYSE THE GENERATED WAV FILE--------------------------------

x = np.fromfile(open('test_sound.wav'),np.int16)[0:100]

plt.plot(x)
plt.show()


#---------------------------------------------------------------------------------------


