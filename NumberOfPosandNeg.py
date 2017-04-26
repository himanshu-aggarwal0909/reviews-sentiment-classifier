import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use('fivethirtyeight')


df  = pd.read_csv('trainingset.csv')
Number_of_positive_sentiments = df['ClassLabel'].sum()
Number_of_neg_sentiments = len(df['ClassLabel']) - Number_of_positive_sentiments

bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(0, Number_of_positive_sentiments, bar_width,
                 alpha=opacity,
                 color='b',
                 
                 error_kw=error_config,
                 label='Positive')

rects2 = plt.bar(1 + bar_width, Number_of_neg_sentiments, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Negative')

plt.xlabel('Sentiments')
plt.ylabel('Number')
plt.title('POS and NEG in reviews')
plt.xticks(np.array([0,1]) + bar_width /2, ('Positive', 'Negative'))
plt.legend()
plt.show()