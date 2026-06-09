## This file contains code for loading and visualizing the results of the Weighted Sum (WS) baseline for multi-objective optimization in multi-task learning.
#  It processes the saved results from multiple runs - 10, 50, 100 and 500 iterations, computes total and average times, and generates 3D scatter plots of the training and test losses. 

## Figure 16 & 17 of the Appendix of the paper.

import pickle
import matplotlib
matplotlib.use('webAgg')
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
np.set_printoptions(precision=5)
from matplotlib.animation import FuncAnimation

def process_times(time_dicts, mode="sum"):
    """
    Compute sum or average of 'total_time' values in a list of dictionaries.

    Parameters:
        time_dicts (list of dict): Each dict has a key 'total_time'.
        mode (str): "sum" or "average"

    Returns:
        float: The computed total or average time
    """
    times = [d.item()['total_time'] for d in time_dicts]
    print("Times:", times)
    
    if mode == "sum":
        return sum(times)
    elif mode == "average":
        return sum(times) / len(times)
    else:
        raise ValueError("mode must be 'sum' or 'average'")



try:
    # Works in normal Python scripts
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for Jupyter/IPython
    FILE_DIR = Path(os.getcwd()).resolve()

file_path500 = FILE_DIR.parent / "Results/UCI_ws_500"
file_path100 = FILE_DIR.parent / "Results/UCI_ws_100"
file_path50 = FILE_DIR.parent / "Results/UCI_ws_50"
file_path10 = FILE_DIR.parent / "Results/UCI_ws_10"

n = 7  # Number of saved results



pref_all500 = []
alpha_all500 = []
train_loss_all500 = []
test_loss_all500 = []

pref_all100 = []
alpha_all100 = []
train_loss_all100 = []
test_loss_all100 = []

pref_all50 = []
alpha_all50 = []
train_loss_all50 = []
test_loss_all50 = []

pref_all10 = []
alpha_all10 = []
train_loss_all10 = []
test_loss_all10 = []


total_time500 = []
total_time100 = []
total_time50 = []
total_time10 = []
for j in range(n):
    file500 = file_path500 / f'first_result_ws{j}.pkl'
    file100 = file_path100 / f'first_result_ws{j}.pkl'
    file50 = file_path50 / f'first_result_ws{j}.pkl'
    file10 = file_path10 / f'first_result_ws{j}.pkl'

    if not os.path.exists(file500):
            # Skip if the file does not exist
            continue
    if not os.path.exists(file100):
            # Skip if the file does not exist
            continue
    if not os.path.exists(file50):
            # Skip if the file does not exist
            continue
    if not os.path.exists(file10):
            # Skip if the file does not exist
            continue
   
    with open(file500, 'rb') as f:
        preference500, alpha500, train_loss500, test_loss500 = pickle.load(f)
    with open(file100, 'rb') as f:
        preference100, alpha100, train_loss100, test_loss100 = pickle.load(f)
    with open(file50, 'rb') as f:
        preference50, alpha50, train_loss50, test_loss50 = pickle.load(f)
    with open(file10, 'rb') as f:
        preference10, alpha10, train_loss10, test_loss10 = pickle.load(f)
        #preference[preference != 0] *= -1 

  
    if  os.path.exists(file500):
        time500 = np.load(file_path500  /f"info_ws{j}.npy", allow_pickle=True)
        pref_all500.append(preference500)
        alpha_all500.append(alpha500)
        train_loss_all500.append(train_loss500)
        test_loss_all500.append(test_loss500)
        total_time500.append(time500)     
  
    if  os.path.exists(file100):
        time100 = np.load(file_path100  /f"info_ws{j}.npy", allow_pickle=True)
        pref_all100.append(preference100)
        alpha_all100.append(alpha100)
        train_loss_all100.append(train_loss100)
        test_loss_all100.append(test_loss100)
        total_time100.append(time100)

    if  os.path.exists(file50):
        time50 = np.load(file_path50  /f"info_ws{j}.npy", allow_pickle=True)
        pref_all50.append(preference50)
        alpha_all50.append(alpha50)
        train_loss_all50.append(train_loss50)
        test_loss_all50.append(test_loss50)
        total_time50.append(time50)
  
    if  os.path.exists(file10):
        time10 = np.load(file_path10  /f"info_ws{j}.npy", allow_pickle=True)
        pref_all10.append(preference10)
        alpha_all10.append(alpha10) 
        train_loss_all10.append(train_loss10)
        test_loss_all10.append(test_loss10)
        total_time10.append(time10)
    

          

pref_all500 = np.array(pref_all500)
alpha_all500 = np.array(alpha_all500)
train_loss_all500 = np.array(train_loss_all500)
test_loss_all500 = np.array(test_loss_all500)

pref_all100 = np.array(pref_all100)
alpha_all100 = np.array(alpha_all100)
train_loss_all100 = np.array(train_loss_all100)
test_loss_all100 = np.array(test_loss_all100)

pref_all50 = np.array(pref_all50)
alpha_all50 = np.array(alpha_all50)
train_loss_all50 = np.array(train_loss_all50)
test_loss_all50 = np.array(test_loss_all50)

pref_all10 = np.array(pref_all10)
alpha_all10 = np.array(alpha_all10)
train_loss_all10 = np.array(train_loss_all10)
test_loss_all10 = np.array(test_loss_all10)



## ***********Total Time************#
Ttime500 = process_times(total_time500, mode="sum")
Atime500 = process_times(total_time500[1:], mode="average")

Ttime100 = process_times(total_time100, mode="sum")
Atime100 = process_times(total_time100[1:], mode="average")

Ttime50 = process_times(total_time50, mode="sum")
Atime50 = process_times(total_time50[1:], mode="average")

Ttime10 = process_times(total_time10, mode="sum")
Atime10 = process_times(total_time10[1:], mode="average")

print("Total Time 500 ",Ttime500)
print("Average Time 500 ",Atime500)
print("Total Time 100 ",Ttime100)
print("Average Time 100 ",Atime100)
print("Total Time 50 ",Ttime50) 
print("Average Time 50 ",Atime50)
print("Total Time 10 ",Ttime10)
print("Average Time 10 ",Atime10)
#print(corr_all)
#print("Test", corr_allt)

#****************** Training set************************
fig = plt.figure(figsize=(10,8),tight_layout = True)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_loss_all500[:,0], train_loss_all500[:,1], train_loss_all500[:,2], label='ws_500', color='brown',s=50)
ax.scatter(train_loss_all100[:,0], train_loss_all100[:,1], train_loss_all100[:,2], label='ws_100', color='g',s=50)
ax.scatter(train_loss_all50[:,0], train_loss_all50[:,1], train_loss_all50[:,2], label='ws_50', color='c',s=50)
ax.scatter(train_loss_all10[:,0], train_loss_all10[:,1], train_loss_all10[:,2], label='ws_10', color='m',s=50)
ax.set_xlabel("age", labelpad=10, fontsize = 20)
ax.set_ylabel("edu", labelpad=10, fontsize = 20)
ax.set_zlabel("ms",labelpad=10, fontsize = 20)
ax.tick_params(axis='x', labelsize=14, length=3)
ax.tick_params(axis='y', labelsize=14, length=3 )
ax.tick_params(axis='z', labelsize=14,  length=3)

ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.zaxis.set_major_locator(mticker.MaxNLocator(6))
ax.zaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.view_init(elev=10, azim=40, roll= -2)
ax.legend(fontsize = 12, loc="best")
#ax.set_title('Objective Space- Training Set WS', fontsize=20)
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()






# Example data
iterations = list(range(1, len(train_loss_all500) + 1))  # Iteration numbers
f1500 = train_loss_all500[:,0]#.tolist()  # Example f1 values
f2500 = train_loss_all500[:,1]#.tolist()  # Example f2 values
f3500 = train_loss_all500[:,2]#.tolist()  # Example f3 

f1100 = train_loss_all100[:,0]#.tolist()  # Example f1 values
f2100 = train_loss_all100[:,1]#.tolist()  # Example f2 values
f3100 = train_loss_all100[:,2]#.tolist()  # Example f3 

f150 = train_loss_all50[:,0]#.tolist()  # Example f1 values
f250 = train_loss_all50[:,1]#.tolist()  # Example f2 values
f350 = train_loss_all50[:,2]#.tolist()  # Example f3        

f110 = train_loss_all10[:,0]#.tolist()  # Example f1 values
f210 = train_loss_all10[:,1]#.tolist()  # Example f2 values
f310 = train_loss_all10[:,2]#.tolist()  # Example f3



# weights corresponding to each fi
"""w1 =  pref_all[:,0]#.tolist()  # Example weights for f1
w2 =  pref_all[:,1]#.tolist()  # Example weights for f2
w3 =  pref_all[:,2]#.tolist()  # Example weights for f3"""


w1 =  alpha_all500[:,0]#.tolist()  # Example weights for f1
w2 =  alpha_all500[:,1]#.tolist()  # Example weights for f2
w3 =  alpha_all500[:,2]#.tolist()  # Example weights for f3


colors = ['r', 'g', 'b']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(iterations, f1500, '-o', color=colors[0], label='age_ws_500', linewidth=0.9)
ax1.plot(iterations, f2500, '-o', color=colors[1], label='edu_ws_500',linewidth=0.9)
ax1.plot(iterations, f3500, '-o', color=colors[2], label='ms_ws_500',linewidth=0.9)

# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1100, '-X', color=colors[0], label='age_ws_100', linewidth=0.9)
ax1.plot(iterations, f2100, '-X', color=colors[1], label='edu_ws_100',linewidth=0.9)
ax1.plot(iterations, f3100, '-X', color=colors[2], label='ms_ws_100',linewidth=0.9)

ax1.plot(iterations, f150, '-^', color=colors[0], label='age_ws_50',linewidth=0.9)
ax1.plot(iterations, f250, '-^', color=colors[1], label='edu_ws_50',linewidth=0.9)
ax1.plot(iterations, f350, '-^', color=colors[2], label='ms_ws_50',linewidth=0.9)

ax1.plot(iterations, f110, '-s', color=colors[0], label='age_ws_10',linewidth=0.9)
ax1.plot(iterations, f210, '-s', color=colors[1], label='edu_ws_10',linewidth=0.9)
ax1.plot(iterations, f310, '-s', color=colors[2], label='ms_ws_10',linewidth=0.9)


ax1.set_ylabel(r'$objectives$', fontsize=14)
ax1.legend()
#ax1.set_title(f"Preference objectives evolution",   fontsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

# --- Lower plot: weights evolution ---
ax2.plot(iterations, w1, '-', color=colors[0], label=r'$\alpha_{age}^*$')
ax2.plot(iterations, w2, '-', color=colors[1], label=r'$\alpha_{edu}^*$')
ax2.plot(iterations, w3, '-', color=colors[2], label=r'$\alpha_{ms}^*$')
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

ax2.set_xlabel(r'Step-$n$', fontsize=14)
ax2.set_ylabel('optimal_weight', fontsize=14)

# --- Styling ---
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
#plt.savefig(f"{file_pathD}\\prefencedtlz_{z}.png", dpi=300)
plt.tight_layout()
plt.show()









#****************** Test set************************

fig = plt.figure(figsize=(10,8),tight_layout = True)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_loss_all500[:,0], test_loss_all500[:,1], test_loss_all500[:,2], label='ws_500', color='brown',s=50)
ax.scatter(test_loss_all100[:,0], test_loss_all100[:,1], test_loss_all100[:,2], label='ws_100', color='g',s=50)
ax.scatter(test_loss_all50[:,0], test_loss_all50[:,1], test_loss_all50[:,2], label='ws_50', color='c',s=50)
ax.scatter(test_loss_all10[:,0], test_loss_all10[:,1], test_loss_all10[:,2], label='ws_10', color='m',s=50)

ax.set_xlabel("age", labelpad=10, fontsize = 20)
ax.set_ylabel("edu", labelpad=10, fontsize = 20)
ax.set_zlabel("ms",labelpad=10, fontsize = 20)
ax.tick_params(axis='x', labelsize=14, length=3)
ax.tick_params(axis='y', labelsize=14, length=3 )
ax.tick_params(axis='z', labelsize=14,  length=3)

ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.zaxis.set_major_locator(mticker.MaxNLocator(6))
ax.zaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.view_init(elev=10, azim=40, roll= -2)
ax.legend(fontsize = 12, loc="best")
#ax.set_title('Objective Space- Test Set', fontsize=20)
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()

#ax_pf.set_title('Objective Space')
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()



# Example data
iterations = list(range(1, len(test_loss_all500) + 1))  # Iteration numbers
f1500 = test_loss_all500[:,0]#.tolist()  # Example f1 values
f2500 = test_loss_all500[:,1]#.tolist()  # Example f2 values
f3500 = test_loss_all500[:,2]#.tolist()  # Example f3 

f1100 = test_loss_all100[:,0]#.tolist()  # Example f1 values
f2100 = test_loss_all100[:,1]#.tolist()  # Example f2 values
f3100 = test_loss_all100[:,2]#.tolist()  # Example f3 

f150 = test_loss_all50[:,0]#.tolist()  # Example f1 values
f250 = test_loss_all50[:,1]#.tolist()  # Example f2 values
f350 = test_loss_all50[:,2]#.tolist()  # Example f3        

f110 = test_loss_all10[:,0]#.tolist()  # Example f1 values
f210 = test_loss_all10[:,1]#.tolist()  # Example f2 values
f310 = test_loss_all10[:,2]#.tolist()  # Example f3




# weights corresponding to each fi
"""w1 =  pref_all[:,0]#.tolist()  # Example weights for f1
w2 =  pref_all[:,1]#.tolist()  # Example weights for f2
w3 =  pref_all[:,2]#.tolist()  # Example weights for f3"""


w1 =  alpha_all500[:,0]#.tolist()  # Example weights for f1
w2 =  alpha_all500[:,1]#.tolist()  # Example weights for f2
w3 =  alpha_all500[:,2]#.tolist()  # Example weights for f3


colors = ['r', 'g', 'b']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1500, '-o', color=colors[0], label='age_ws_500', linewidth=0.9)
ax1.plot(iterations, f2500, '-o', color=colors[1], label='edu_ws_500',linewidth=0.9)
ax1.plot(iterations, f3500, '-o', color=colors[2], label='ms_ws_500',linewidth=0.9)

# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1100, '-X', color=colors[0], label='age_ws_100', linewidth=0.9)
ax1.plot(iterations, f2100, '-X', color=colors[1], label='edu_ws_100',linewidth=0.9)
ax1.plot(iterations, f3100, '-X', color=colors[2], label='ms_ws_100',linewidth=0.9)

ax1.plot(iterations, f150, '-^', color=colors[0], label='age_ws_50',linewidth=0.9)
ax1.plot(iterations, f250, '-^', color=colors[1], label='edu_ws_50',linewidth=0.9)
ax1.plot(iterations, f350, '-^', color=colors[2], label='ms_ws_50',linewidth=0.9)

ax1.plot(iterations, f110, '-s', color=colors[0], label='age_ws_10',linewidth=0.9)
ax1.plot(iterations, f210, '-s', color=colors[1], label='edu_ws_10',linewidth=0.9)
ax1.plot(iterations, f310, '-s', color=colors[2], label='ms_ws_10',linewidth=0.9)

ax1.set_ylabel(r'$objectives$', fontsize=14)
ax1.legend()
#ax1.set_title(f"Preference objectives evolution",   fontsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

# --- Lower plot: weights evolution ---
ax2.plot(iterations, w1, '-', color=colors[0], label=r'$\alpha_{age}^*$')
ax2.plot(iterations, w2, '-', color=colors[1], label=r'$\alpha_{edu}^*$')
ax2.plot(iterations, w3, '-', color=colors[2], label=r'$\alpha_{ms}^*$')
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

ax2.set_xlabel(r'Step-$n$', fontsize=14)
ax2.set_ylabel('optimal_weight', fontsize=14)

# --- Styling ---
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
#plt.savefig(f"{file_pathD}\\prefencedtlz_{z}.png", dpi=300)
plt.tight_layout()
plt.show()



