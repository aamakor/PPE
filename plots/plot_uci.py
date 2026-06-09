# This file contains code for loading and visualizing the results of the Preference Pareto Exploration
# (PPE) framework for multi-objective optimization in multi-task learning on the 3-task UCI dataset (Figure 5)

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


# Function to identify suboptimal points that are not on the current front
def is_not_dominated(K, b):
    """
    Check if vector b is not dominated by any row in matrix K.

    Args:
        K: 2D numpy array of shape (n,3), where each column is a solution.
        b: 1D numpy array of shape (3,), representing the candidate solution.
    Returns:
        True if b is not dominated by any row in K, False otherwise.
    """
    for k in K:
        # Check if k dominates b
        if np.all(k <= b) and np.any(k < b):
            #print(f"Vector {b} is dominated by {k}")
            return False  # b is dominated by k
        
    return True  # b is not dominated by any row in K

#Initial Point:  [0.43292 0.21145 0.02439]
try:
    # Works in normal Python scripts
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for Jupyter/IPython
    FILE_DIR = Path(os.getcwd()).resolve()

file_path = FILE_DIR.parent / "Results/UCI"

n = 10

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
        preferenceA, alphas = preferenceA[0], alphas[0]
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
    
    
preference = np.array(pref_all)
prefA_all = np.array(prefA_all)
alpha_all = np.array(alpha_all)
predictor_point = np.array(pred_all)       
corrector_point = np.array(corr_all)
preft_all = np.array(preft_all)
predictor_pointt = np.array(predt_all)       
corrector_pointt = np.array(corrt_all)


final_preference = []
final_opt_pred = []
final_opt_corr = []

all_subopt_points = np.concatenate((np.array(corr_all),initial_point[-1, :].reshape(1,3)),axis=0)

for i in range(len(all_subopt_points)):
    b = all_subopt_points[i]
    if is_not_dominated(all_subopt_points, b):
        idx = np.where((np.array(corr_all) == b).all(axis=1))[0]
        if len(idx) > 0:
            final_preference.append(np.array(pref_all)[idx[0]])
            final_opt_pred.append(np.array(pred_all)[idx[0]])
            final_opt_corr.append(np.array(corr_all)[idx[0]])



final_preferencet = []
final_opt_predt = []
final_opt_corrt = []

all_subopttest_points = np.concatenate((np.array(corrt_all),initialt_point[-1, :].reshape(1,3)),axis=0)

for i in range(len(all_subopttest_points)):
    bt = all_subopttest_points[i]
    if is_not_dominated(all_subopttest_points, bt):
        idx = np.where((np.array(corrt_all) == bt).all(axis=1))[0]
        if len(idx) > 0:
            final_preferencet.append(np.array(preft_all)[idx[0]])
            final_opt_predt.append(np.array(predt_all)[idx[0]])
            final_opt_corrt.append(np.array(corrt_all)[idx[0]])
    

    
preferencer = np.array(final_preference)
predictor_pointr = np.array(final_opt_pred)   
corrector_pointr = np.array(final_opt_corr)
preferencetr = np.array(final_preferencet)
predictor_pointtr = np.array(final_opt_predt)
corrector_pointtr = np.array(final_opt_corrt)

## Identifying optimal and suboptimal points that are not on the Pareto front for visualization purposes. These are the points that are generated during the corrector steps but do not lie on the final Pareto front. 
# We will plot these points in a different color to distinguish them from the optimal points on the Pareto front.
mask_common = np.any(np.all(corrector_point[:, None] == corrector_pointr[None, :], axis=2), axis=1)
mask_unique = ~mask_common
mask_commont = np.any(np.all(corrector_pointt[:, None] == corrector_pointtr[None, :], axis=2), axis=1)
mask_uniquet = ~mask_commont


## Scaling for better visualization as shown in figure 5a of the paper using z-score normalization (standardization) for both training and test points. 
# This will help in visualizing the points more clearly in the 2D plot. We remove the initial point from the scaling and only scale the subsequent points for better visualization of the trajectory.

predictor_points = (predictor_point - predictor_point.mean(axis=0)) / predictor_point.std(axis=0)
corrector_points = (corrector_point - corrector_point.mean(axis=0))  / corrector_point.std(axis=0)  
predictor_pointts = (predictor_pointt - predictor_pointt.mean(axis=0)) / predictor_pointt.std(axis=0) 
corrector_pointts = (corrector_pointt - corrector_pointt.mean(axis=0))  / corrector_pointt.std(axis=0)  

predictor_pointrs = (predictor_pointr - predictor_pointr.mean(axis=0)) / predictor_pointr.std(axis=0) 
corrector_pointrs = (corrector_pointr - corrector_pointr.mean(axis=0))  / corrector_pointr.std(axis=0) 
predictor_pointtrs = (predictor_pointtr - predictor_pointtr.mean(axis=0)) / predictor_pointtr.std(axis=0) 
corrector_pointtrs = (corrector_pointtr - corrector_pointtr.mean(axis=0))  / corrector_pointtr.std(axis=0)  

allc   = corrector_point[mask_unique]
allct   = corrector_pointt[mask_uniquet]




## Plot the train scaled predicted and corrected points with preferences for each step

iterations = list(range(1, len(predictor_points) + 1))  # Iteration numbers
iterations = np.array(iterations)
f1 = predictor_points[:,0]
f2 = predictor_points[:,1]
f3 = predictor_points[:,2]

f1_corr = corrector_points[:,0]
f2_corr = corrector_points[:,1]
f3_corr = corrector_points[:,2]

mask_col1 = np.isin(corrector_point[:,0], corrector_pointr[:,0])
mask_col2 = np.isin(corrector_point[:,1], corrector_pointr[:,1])
mask_col3 = np.isin(corrector_point[:,2], corrector_pointr[:,2])

mask_col1 = np.asarray(mask_col1).flatten().astype(bool)
mask_col2 = np.asarray(mask_col2).flatten().astype(bool)
mask_col3 = np.asarray(mask_col3).flatten().astype(bool)

# weights corresponding to each fi
w1 =  preference[:,0]#.tolist()  # Example weights for f1
w2 =  preference[:,1]#.tolist()  # Example weights for f2
w3 =  preference[:,2]#.tolist()  # Example weights for f3

colors = ['r', 'g', 'b']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1, '-o', color=colors[0], label='age_pred', linewidth=0.9)
ax1.plot(iterations, f2, '-o', color=colors[1], label='edu_pred',linewidth=0.9)
ax1.plot(iterations, f3, '-o', color=colors[2], label='ms_pred',linewidth=0.9)


ax1.plot(iterations, f1_corr, '--x', color=colors[0], label='age_corr',linewidth=0.9)
ax1.plot(iterations, f2_corr, '--x', color=colors[1], label='edu_corr',linewidth=0.9)
ax1.plot(iterations, f3_corr, '--x', color=colors[2], label='ms_corr',linewidth=0.9)

ax1.plot(iterations[mask_col1], f1_corr[mask_col1], '^', color='black',label = "optimal point", linewidth=0.9)
ax1.plot(iterations[mask_col2], f2_corr[mask_col2], '^', color='black',linewidth=0.9)
ax1.plot(iterations[mask_col3], f3_corr[mask_col3], '^',color='black',linewidth=0.9)

ax1.set_ylabel(r'$objectives$', fontsize=14)
ax1.legend()
#ax1.set_title(f"Preference Objective",   fontsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))


# --- Lower plot: weights evolution ---
ax2.plot(iterations, w1, '-', color=colors[0], label=r'$\pi_{age}$')
ax2.plot(iterations, w2, '-', color=colors[1], label=r'$\pi_{edu}$')
ax2.plot(iterations, w3, '-', color=colors[2], label=r'$\pi_{ms}$')
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







fig = plt.figure(figsize=(10,8),tight_layout = True)
ax = fig.add_subplot(111, projection='3d')

for j in range(len(preference)):
    
    if j == 0 :
        print("Initial point, ", initial_point[-1])
        print("Pred ", j, ": ", predictor_point)
        
        ax.scatter(initial_point[-1][0], initial_point[-1][1], initial_point[-1][2], label='Initial Point', color='r',s=50)
        #Uncomment below for interactive plot of train set points
    """    #ax.scatter(initial_point[-1,0], initial_point[-1,1], initial_point[-1,2], label='Initial Point', color='r',s=100)
        ax.plot([initial_point[-1][0], predictor_point[j ,0]], [initial_point[-1][1], predictor_point[j ,1]], [initial_point[-1][2], predictor_point[j ,2]],"-o", label='predictor step', color='g',linewidth=2)
        ax.plot([predictor_point[j ,0], corrector_point[j, 0]], [predictor_point[j, 1], corrector_point[j,1]], [predictor_point[j,2], corrector_point[j,2]],"-o", label='corrector Step', color='b',linewidth=2)
    if j> 0:
        ax.plot([corrector_pointold[0], predictor_point[j][0]], [corrector_pointold[1],predictor_point[j][1] ], [corrector_pointold[2], predictor_point[j][2]],"-o", color='g',linewidth=2)
        ax.plot([predictor_point[j,0], corrector_point[j,0]], [predictor_point[j,1], corrector_point[j,1]], [predictor_point[j,2], corrector_point[j,2]],"-o",  color='b',linewidth=2)

    print("Pref ", j, ": ", preference[j])
    print("Pred ", j, ": ", predictor_point[j])
    print("Corr ", j, ": ", corrector_point[j])

    corrector_pointold= corrector_point[j,:]"""



ax.scatter(corrector_pointr[:,0], corrector_pointr[:,1], corrector_pointr[:,2],marker='^', label='optimal points', color='black',s=50)
ax.scatter(allc[:,0], allc[:,1], allc[:,2], label='corrector points', color='m',s=50)

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






#****************** Test set************************

fig = plt.figure(figsize=(10,8),tight_layout = True)
ax = fig.add_subplot(111, projection='3d')


for j in range(len(preferencet)):
    
    if j == 0 :
        print("Initial test point ", initialt_point[-1])
        
        ax.scatter(initialt_point[-1][0], initialt_point[-1][1], initialt_point[-1][2], label='Initial Point', color='r',s=50)
        #Uncomment below for interactive plot of test set points
    """    ax.plot([initialt_point[-1][0], predictor_pointt[j ,0]], [initialt_point[-1][1], predictor_pointt[j ,1]], [initialt_point[-1][2], predictor_pointt[j ,2]],"-o", label='Predictor', color='g',linewidth=2)
        ax.plot([predictor_pointt[j ,0], corrector_pointt[j, 0]], [predictor_pointt[j, 1], corrector_pointt[j,1]], [predictor_pointt[j,2], corrector_pointt[j,2]],"-o", label='Corrector', color='b',linewidth=2)
    if j> 0:
        ax.plot([correctort_pointold[0], predictor_pointt[j][0]], [correctort_pointold[1],predictor_pointt[j][1] ], [correctort_pointold[2], predictor_pointt[j][2]],"-o", color='g',linewidth=2)
        ax.plot([predictor_pointt[j,0], corrector_pointt[j,0]], [predictor_pointt[j,1], corrector_pointt[j,1]], [predictor_pointt[j,2], corrector_pointt[j,2]],"-o",     color='b',linewidth=2)
   
    print("Pref ", j, ": ", preferencet[j])
    print("Pred ", j, ": ", predictor_pointt[j])
    print("Corr ", j, ": ", corrector_pointt[j])"""

    #print("Preft ", j, ": ", preferencet[j])
    #print("Predt", j, ": ", predictor_pointt[j])
    #print("Corrt ", j, ": ", corrector_pointt[j])

    #correctort_pointold= corrector_pointt[j,:]
    # Pause to allow plot to update
    #plt.pause(0.6)

    """t_along_line = 1.0 # Adjust this value (0.0 to 1.0) to place the label along the line

    label_x = initial_point[-1,0] + t_along_line * (predictor_point[j,0] - initial_point[-1,0])
    label_y = initial_point[-1,1] + t_along_line * (predictor_point[j,1] - initial_point[-1,1])
    label_z = initial_point[-1,2] + t_along_line * (predictor_point[j,2] - initial_point[-1,2])

    # Add the text annotation
    ax.text(label_x, label_y, label_z, f"{preference[j]}",color="black",fontsize=11, ha='center',va='bottom',zdir=None, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))"""

ax.scatter(corrector_pointtr[:,0], corrector_pointtr[:,1], corrector_pointtr[:,2], marker = '^',label='optimal points', color='black',s=50)
ax.scatter(allct[:,0], allct[:,1], allct[:,2], label='corrector points', color='m',s=50)
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
ax.view_init(elev=10, azim=40, roll = -2)
ax.legend(fontsize = 12, loc="best")
#ax_pf.set_title('Objective Space')
#plt.savefig("pareto_frontxcalldxcqs13.png", dpi=300)
ax.grid(True)
plt.show()

