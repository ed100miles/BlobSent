from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import datetime as dt
import numpy as np
import matplotlib.dates as md
import dateutil
from scipy.ndimage.filters import gaussian_filter1d

style.use('fivethirtyeight')
#'fivethirtyeight', seaborn-pastel, dark_background
csvfile = 'tweet_out.csv'

graph_data = open(csvfile, 'r').read()
lines = graph_data.split('\n')

xs = []
ys_mean = []

bias_skew = 0.2
smoothing = 250

for line in lines[200:]:
    if len(line) > 1:
        x, y_mean, pos, neg, sent  = line.split(',')
        x_dt = dt.datetime.utcfromtimestamp(float(x))
        xs.append(x_dt)
        ys_mean.append(((float(y_mean))-bias_skew)*100)
        
ys_mean = np.asarray(ys_mean)
ys_mean_smooth = gaussian_filter1d(ys_mean, sigma=smoothing)

plt.plot_date(xs, ys_mean_smooth, '-', linewidth=2, label='Mean Sentiment')

plt.fill_between(xs, ys_mean_smooth, 0, where=(ys_mean_smooth>0), facecolor='g', alpha=0.5)
plt.fill_between(xs, ys_mean_smooth, 0, where=(ys_mean_smooth<0), facecolor='r', alpha=0.5)

plt.subplots_adjust(bottom=0.20)
plt.xticks(rotation=45)
plt.xlabel('Time (UK)')
plt.ylabel('Sentiment %')
plt.title('Joe Biden Twitter Sentiment')
plt.legend(loc='best')

plt.show()

