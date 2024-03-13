import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_ex3_2.csv", index_col=0)

grouped = df.groupby('camber')
"""
for name, group in grouped:
    

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
    plt.subplot(2, 2, 1)
    plt.plot(group["angle"], group["fd"])
    plt.title("Fd")

    plt.subplot(2, 2, 2)
    plt.plot(group["angle"], group["fl"])
    plt.title("Fl")

    plt.subplot(2, 2, 3)
    plt.plot(group["angle"], group["fl"] / group["fd"])
    plt.title("Fl / Fd")

    plt.subplot(2, 2, 4)
    plt.plot(group["angle"], group["j"])
    plt.title("J")
    
    fig.suptitle(name)

    plt.show(block=False)
"""


names = grouped.groups.keys()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
plt.subplot(2, 2, 1)
for name, group in grouped:
    plt.plot(group["angle"], group["fd"])
plt.legend(names)
    
plt.title("Fd")
plt.subplot(2, 2, 2)
for name, group in grouped:
    plt.plot(group["angle"], group["fl"])
plt.legend(names)
    
plt.title("Fl")
plt.subplot(2, 2, 3)
for name, group in grouped:
    plt.plot(group["angle"], group["fl"] / group["fd"])
plt.legend(names)
    
plt.title("Fl / Fd")
plt.subplot(2, 2, 4)
for name, group in grouped:
    plt.plot(group["angle"], group["j"])
plt.legend(names)
    
plt.title("J")

fig.suptitle("All cambers")

plt.show()



