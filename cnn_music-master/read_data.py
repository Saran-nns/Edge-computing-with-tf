import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
headers = 'ROW WISE ACCESS TO THE DATA'
df = pd.read_csv('MYDATA',names='ROW1')
print (df)

x = 'TIME FRAME OF THE DATA. PROBABLY 2 SECONDS'  
y = 'Amplitutes'

# plot
plt.plot(x,y)
# beautify the x-labels
plt.gcf().autofmt_xdate()
plt.show()
