## This file contains code for loading and visualizing the results of the Preference Pareto Exploration (PPE) framework 
# compared to either the Pareto MTL or Weighted Sum (WS) baseline for multi-objective optimization in multi-task learning.
# Section 4.2 and 4.3  of the paper where we minimized 2 out of 3 objectives in 6 steps. 
# Totally 7 optimal points (including the initial point) are obtained for each method.
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


method = "pareto_mtl" # "ws" for Weighted Sum baseline, "pareto_mtl" for Pareto MTL.



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

file_pathD = FILE_DIR.parent / "Results/UCI_1&3"
if method == "pareto_mtl":
    #file_path = FILE_DIR.parent / "Results/UCI_pareto_mtl_50"
    file_path = FILE_DIR.parent / "Results/UCI_pareto_mtl_100"
elif method == "ws":
    file_path = FILE_DIR.parent / "Results/UCI_ws_100"



n = 15  # Number of saved results
pref_all = []
alpha_all = []
train_loss_all = []
test_loss_all = []


pref_all5 = []
alpha_all5 = []
train_loss_all5 = []
test_loss_all5 = []

corr_all = []
corr_allt = []

total_time = []
total_time5 = []
for j in range(n):
    #file = file_path / f'first_result_ws{j}.pkl'
    file = file_path / f'first_result_pareto_mtl{j}.pkl' if method == "pareto_mtl" else file_path / f'first_result_ws{j}.pkl'
    
    
    if not os.path.exists(file):
            # Skip if the file does not exist
            continue

   
    with open(file, 'rb') as f:
        preference, alpha, train_loss, test_loss = pickle.load(f)

        #preference[preference != 0] *= -1 

    if j < 6:
        file_cm = file_pathD / f'first_result_cen{j}.pkl'
        file_cmt = file_pathD / f'first_result_test_cen{j}.pkl'
        if not os.path.exists(file_cm):
                # Skip if the file does not exist
                continue
        if not os.path.exists(file_cmt):
                # Skip if the file does not exist
                continue

        with open(file_cm, 'rb') as f:
            initial_point,_,corrector_point,_ = pickle.load(f)
            corrector_point =corrector_point[0]
        
        with open(file_cmt, 'rb') as f:
            initial_pointt,_,corrector_pointt, _ = pickle.load(f)
            corrector_pointt = corrector_pointt[0]
        
        if j == 0:
            corr_all.append(initial_point[0])
            corr_allt.append(initial_pointt[0])
        
        corr_all.append(corrector_point)
        corr_allt.append(corrector_pointt)
    
      
    
    if  os.path.exists(file):
        #time = np.load(file_path  /f"info_ws{j}.npy", allow_pickle=True)
        time = np.load(file_path  /f"info_pareto_mtl{j}.npy", allow_pickle=True) if method == "pareto_mtl" else np.load(file_path  /f"info_ws{j}.npy", allow_pickle=True)
        print(time)

        pref_all.append(preference)
        alpha_all.append(alpha)
        train_loss_all.append(train_loss)
        test_loss_all.append(test_loss)
        total_time.append(time)

  


pref_all = np.array(pref_all)
alpha_all = np.array(alpha_all)
train_loss_all = np.array(train_loss_all)
test_loss_all = np.array(test_loss_all)
corr_all = np.array(corr_all)
corr_allt = np.array(corr_allt) 

pref_all5 = np.array(pref_all5)
alpha_all5 = np.array(alpha_all5)
train_loss_all5 = np.array(train_loss_all5)
test_loss_all5 = np.array(test_loss_all5)


#print(len(train_loss_all))
#print(len(corr_all))

## ***********Total Time************#
Ttime = process_times(total_time, mode="sum")
Atime = process_times(total_time[1:], mode="average")

print("Total Time ParetoMTL ",Ttime)
print("Average Time ParetoMTL ",Atime)

#print(corr_all)
#print("Test", corr_allt)

#****************** Training set************************
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_loss_all[:,0], train_loss_all[:,1], train_loss_all[:,2], label='pareto_mtl' if method == "pareto_mtl" else 'ws', color='g',s=50)
ax.scatter(corr_all[:,0], corr_all[:,1], corr_all[:,2], label='ppe', color='m',s=50)
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
ax.view_init(elev=30, azim=60)
ax.legend(fontsize = 12, loc="best")
ax.set_title('PPE versus ' + ('ParetoMTL' if method == "pareto_mtl" else 'WS')+ ' (train set)', fontsize=20)
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()









# Example data
iterations = list(range(1, len(train_loss_all) + 1))  # Iteration numbers

f1 = train_loss_all[:,0]#.tolist()  # Example f1 values
f2 = train_loss_all[:,1]#.tolist()  # Example f2 values
f3 = train_loss_all[:,2]#.tolist()  # Example f3 



f1_cm = corr_all[:,0]#.tolist()  # Example f1 values
f2_cm = corr_all[:,1]#.tolist()  # Example f2 values
f3_cm = corr_all[:,2]#.tolist()  # Example f3 




# weights corresponding to each fi
"""w1 =  pref_all[:,0]#.tolist()  # Example weights for f1
w2 =  pref_all[:,1]#.tolist()  # Example weights for f2
w3 =  pref_all[:,2]#.tolist()  # Example weights for f3"""


w1 =  alpha_all[:,0]#.tolist()  # Example weights for f1
w2 =  alpha_all[:,1]#.tolist()  # Example weights for f2
w3 =  alpha_all[:,2]#.tolist()  # Example weights for f3


colors = ['r', 'g', 'b']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1, '-X', color=colors[0], label='age_' + ('pareto_mtl' if method == "pareto_mtl" else 'ws'), linewidth=0.9)
ax1.plot(iterations, f2, '-X', color=colors[1], label='edu_' + ('pareto_mtl' if method == "pareto_mtl" else 'ws'), linewidth=0.9)
ax1.plot(iterations, f3, '-X', color=colors[2], label='ms_' + ('pareto_mtl' if method == "pareto_mtl" else 'ws'), linewidth=0.9)



ax1.plot(iterations, f1_cm, '--^', color=colors[0], label='age_ppe',linewidth=0.9)
ax1.plot(iterations, f2_cm, '--^', color=colors[1], label='edu_ppe',linewidth=0.9)
ax1.plot(iterations, f3_cm, '--^', color=colors[2], label='ms_ppe',linewidth=0.9)
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

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_loss_all[:,0], test_loss_all[:,1], test_loss_all[:,2], label='ws' if method == "ws" else 'pareto_mtl', color='c',s=50)
ax.scatter(corr_allt[:,0], corr_allt[:,1], corr_allt[:,2], label='ppe', color='m',s=50)
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
ax.view_init(elev=30, azim=60)
ax.legend(fontsize = 12, loc="best")
ax.set_title('PPE versus ' + ('ParetoMTL' if method == "pareto_mtl" else 'WS')+ '(test set)', fontsize=20)
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()

#ax_pf.set_title('Objective Space')
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()



# Example data
iterations = list(range(1, len(test_loss_all) + 1))  # Iteration numbers
f1 = test_loss_all[:,0]#.tolist()  # Example f1 values
f2 = test_loss_all[:,1]#.tolist()  # Example f2 values
f3 = test_loss_all[:,2]#.tolist()  # Example f3 


f1_cm = corr_allt[:,0]#.tolist()  # Example f1 values
f2_cm = corr_allt[:,1]#.tolist()  # Example f2 values
f3_cm = corr_allt[:,2]#.tolist()  # Example f3 




# weights corresponding to each fi
"""w1 =  pref_all[:,0]#.tolist()  # Example weights for f1
w2 =  pref_all[:,1]#.tolist()  # Example weights for f2
w3 =  pref_all[:,2]#.tolist()  # Example weights for f3"""


w1 =  alpha_all[:,0]#.tolist()  # Example weights for f1
w2 =  alpha_all[:,1]#.tolist()  # Example weights for f2
w3 =  alpha_all[:,2]#.tolist()  # Example weights for f3


colors = ['r', 'g', 'b']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1, '-X', color=colors[0], label='age_' + ('pareto_mtl' if method == "pareto_mtl" else 'ws'), linewidth=0.9)
ax1.plot(iterations, f2, '-X', color=colors[1], label='edu_' + ('pareto_mtl' if method == "pareto_mtl" else 'ws'), linewidth=0.9)
ax1.plot(iterations, f3, '-X', color=colors[2], label='ms_' + ('pareto_mtl' if method == "pareto_mtl" else 'ws'), linewidth=0.9)


ax1.plot(iterations, f1_cm, '--^', color=colors[0], label='age_ppe',linewidth=0.9)
ax1.plot(iterations, f2_cm, '--^', color=colors[1], label='edu_ppe',linewidth=0.9)
ax1.plot(iterations, f3_cm, '--^', color=colors[2], label='ms_ppe',linewidth=0.9)
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



