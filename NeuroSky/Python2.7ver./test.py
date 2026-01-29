import matplotlib.pyplot as plt
from random import random

x = []
y = []
res = 0
# def do_something():
#     for p in range(10000000):
#         res += p


fig, ax = plt.subplots()

for i in range(50):
    x.append(i)
    y.append(50 * random())
    ax.cla()  # clear plot
    ax.plot(x, y, 'r', lw=1)  # draw line chart
    # ax.bar(y, height=y, width=0.3) # draw bar chart
    # do_something()
    plt.pause(0.1)