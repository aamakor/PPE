# This code is for plotting the results of the MultiMNIST experiment saved as INT, as shown in figure 4 of the paper.
#  It reads the saved results from the specified directory, processes the data, and creates the plots for both training and test sets.
import matplotlib
matplotlib.use('webAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.ticker as mticker




from pathlib import Path
import os
import pickle


try:
    # Works in normal Python scripts
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for Jupyter/IPython
    FILE_DIR = Path(os.getcwd()).resolve()

file_path = FILE_DIR.parent / "Results/INT"

n = 30

pref_all = []
corr_all = []
alpha_all = []
prefA_all = []
pred_all = []
preft_all = []
corrt_all = []
predt_all = []

for j in range(n):
    file = file_path / f'first_result_cen{j}.pkl'
    file_alpha = file_path / f'first_alphas_cen{j}.pkl'
    file_t = file_path / f'first_result_test_cen{j}.pkl'
    if not os.path.exists(file):
            # Skip if the file does not exist
            continue
    if not os.path.exists(file_alpha):
            # Skip if the file does not exist
            continue
    if not os.path.exists(file_t):
            # Skip if the file does not exist
            continue
    with open(file, 'rb') as f:
        initial_point,predictor_point,corrector_point, preference = pickle.load(f)
        predictor_point,corrector_point = predictor_point[0],corrector_point[0]
        #preference[preference != 0] *= -1 

    with open(file_t, 'rb') as f:
        initialt_point,predictort_point,correctort_point, preferencet = pickle.load(f)
        predictort_point,correctort_point = predictort_point[0],correctort_point[0]
        #preferencet[preferencet != 0] *= -1 

    with open(file_alpha, 'rb') as f:
        preferenceA , alphas= pickle.load(f)
        preferenceA, alphas = preferenceA[0], alphas[1]
        #preferenceA[preferenceA != 0] *= -1 
    
    if  os.path.exists(file):
        pref_all.append(preference[-1])
        pred_all.append(predictor_point)
        corr_all.append(corrector_point)

    if  os.path.exists(file_t):
        preft_all.append(preferencet[-1])
        predt_all.append(predictort_point)   
        corrt_all.append(correctort_point)   

    if  os.path.exists(file_alpha):
        prefA_all.append(preferenceA[-1])
        alpha_all.append(alphas)   
    
    
pref_all = np.array(pref_all)
prefA_all = np.array(prefA_all)
alpha_all = np.array(alpha_all)
pred_all = np.array(pred_all)       
corr_all = np.array(corr_all)
preft_all = np.array(preft_all)
predt_all = np.array(predt_all)       
corrt_all = np.array(corrt_all)



all_subopt_points = np.concatenate((initial_point[-1, :].reshape(1,3), corr_all), axis=0)
all_subopttest_points = np.concatenate((initialt_point[-1, :].reshape(1,3), corrt_all), axis=0)


## Scaling for better visualization as shown in figure 4a of the paper using z-score normalization (standardization) for both training and test points. 
# This will help in visualizing the points more clearly in the 2D plot. We remove the initial point from the scaling and only scale the subsequent points for better visualization of the trajectory.
preference = pref_all
predictor_point = (all_subopt_points[1:] - all_subopt_points[1:].mean(axis=0)) / all_subopt_points[1:].std(axis=0) 
corrector_point = (all_subopt_points[1:] - all_subopt_points[1:].mean(axis=0))  / all_subopt_points[1:].std(axis=0) 
preferencet = preft_all
predictor_pointt = (all_subopttest_points[1:] - all_subopttest_points[1:].mean(axis=0)) / all_subopttest_points[1:].std(axis=0) 
corrector_pointt = (all_subopttest_points[1:] - all_subopttest_points[1:].mean(axis=0))  / all_subopttest_points[1:].std(axis=0) 


iterations = list(range(1, len(predictor_point) + 1))  # Iteration numbers
f1 = predictor_point[:,0]
f2 = predictor_point[:,1]
f3 = predictor_point[:,2]
f1_corr = corrector_point[:,0]
f2_corr = corrector_point[:,1]
f3_corr = corrector_point[:,2]
# weights corresponding to each fi
w1 =  preference[:,0]
w2 =  preference[:,1]
w3 =  preference[:,2]

colors = ['r', 'g', 'b']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1, '-o', color=colors[0], label='fmnist_pred', linewidth=0.9)
ax1.plot(iterations, f2, '-o', color=colors[1], label='kmnist_pred',linewidth=0.9)
ax1.plot(iterations, f3, '-o', color=colors[2], label='mnist_pred',linewidth=0.9)


ax1.plot(iterations, f1_corr, '--x', color=colors[0], label='fmnist_corr',linewidth=0.9)
ax1.plot(iterations, f2_corr, '--x', color=colors[1], label='kmnist_corr',linewidth=0.9)
ax1.plot(iterations, f3_corr, '--x', color=colors[2], label='mnist_corr',linewidth=0.9)

ax1.set_ylabel(r'$objectives$', fontsize=14)
ax1.legend()
#ax1.set_title(f"Preference Objective",   fontsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))


# --- Lower plot: weights evolution ---
ax2.plot(iterations, w1, '-', color=colors[0], label=r'$\pi_{fmnist}$')
ax2.plot(iterations, w2, '-', color=colors[1], label=r'$\pi_{kmnist}$')
ax2.plot(iterations, w3, '-', color=colors[2], label=r'$\pi_{mnist}$')
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

ax2.set_xlabel(r'Step-$n$', fontsize=14)
ax2.set_ylabel('Preferences', fontsize=14)

# --- Styling ---
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
#plt.savefig(f"{file_pathD}\\prefencedtlz_{z}.png", dpi=300)
plt.tight_layout()
plt.show()



## FIGURE 4b and 4c below: 3D plot of the objective space for training and test points respectively.


fig = plt.figure(figsize=(10,8),tight_layout = True)
ax = fig.add_subplot(111, projection='3d')
hxs = all_subopt_points[:, 0]
mys = all_subopt_points[:, 1] 
wzs = all_subopt_points[:, 2]
gears = list(range(0, len(pred_all)+1))


ax.plot(hxs, mys, wzs, '-',
        color='#FF69B4', linewidth=4.5, zorder=5)

for i in range(len(hxs)):
    if i == 0:
        color = '#FF0000' 
        edge_color = '#CC0000'
    
        # Big circle marker
        ax.scatter(hxs[i]+0.00008, mys[i]-0.00008, wzs[i]-0.00008,
                s=200, color=color, edgecolor=edge_color, linewidth=3, label='initial point', zorder=5)

# Connected line with big circle markers ("-o" style)
ax.plot(hxs[1:], mys[1:], wzs[1:], '-o',
        color='#FF69B4',          # pink line
        linewidth=3,
        markersize=11,
        markerfacecolor="#FFB6C1",#'white',
        markeredgecolor='#DB7093',
        markeredgewidth=3,label='optimal point',
        zorder=5)

for i in range(1,len(hxs)):
    ax.text(hxs[i], mys[i], wzs[i],  str(gears[i]),
            fontsize=9, fontweight='bold', color='#DB7093',
            ha='center', va='center', zorder=8,
            bbox=dict(boxstyle="circle,pad=0.52",
                      facecolor="#FFB6C1",
                      edgecolor="#DB7093",
                      linewidth=3, label='optimal point with Step'))
    


# ==================== Styling ====================
ax.set_xlabel("fmnist",labelpad=10, fontsize = 20)
ax.set_ylabel("kmnist",labelpad=10, fontsize = 20)
ax.set_zlabel("mnist",labelpad = 10,fontsize = 20)
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
#ax.set_title('Objective Space- Training Set', fontsize=20)
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()


## Test code for plotting the test set results (similar to above, but with test data)


fig = plt.figure(figsize=(10,8),tight_layout = True)
ax = fig.add_subplot(111, projection='3d')
hxs = all_subopttest_points[:, 0] 
mys = all_subopttest_points[:, 1] 
wzs = all_subopttest_points[:, 2] 
gears = list(range(0, len(pred_all)+1)) 



ax.plot(hxs, mys, wzs, '-',
        color='#FF69B4', linewidth=4.5, zorder=5)

for i in range(len(hxs)):
    if i == 0:
        color = '#FF0000' 
        edge_color = '#CC0000'
    
        ax.scatter(hxs[i]+0.00008, mys[i]-0.00008, wzs[i]-0.00008,
                s=200, color=color, edgecolor=edge_color, linewidth=3, label='initial point', zorder=5)


ax.plot(hxs[1:], mys[1:], wzs[1:], '-o',
        color='#FF69B4',          # pink line
        linewidth=3,
        markersize=11,
        markerfacecolor="#FFB6C1",#'white',
        markeredgecolor='#DB7093',
        markeredgewidth=3,label='optimal point',
        zorder=5)

for i in range(1,len(hxs)):
    ax.text(hxs[i], mys[i], wzs[i],  str(gears[i]),
            fontsize=9, fontweight='bold', color='#DB7093',
            ha='center', va='center', zorder=8,
            bbox=dict(boxstyle="circle,pad=0.52",
                      facecolor="#FFB6C1",
                      edgecolor="#DB7093",
                      linewidth=3, label='optimal point with Step'))
    

# ==================== Styling ====================
ax.set_xlabel("fmnist",labelpad=10, fontsize = 20)
ax.set_ylabel("kmnist",labelpad=10, fontsize = 20)
ax.set_zlabel("mnist",labelpad = 10,fontsize = 20)
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
#ax.set_title('Objective Space- Training Set', fontsize=20)
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()
