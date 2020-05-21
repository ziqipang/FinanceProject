import matplotlib.pyplot as plt
import matplotlib
import os

f = plt.figure()
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=10)
plt.bar(['2', '10', '12', '15'], [0.288, 0.289, 0.09, 0.334], tick_label=['2', '10', '12', '15'])
plt.xticks(rotation=270)
plt.xlabel('group')
plt.ylabel('weight')
plt.title("weights of each group")
f.savefig(os.path.join(os.getcwd(), '../figs/result.pdf'))