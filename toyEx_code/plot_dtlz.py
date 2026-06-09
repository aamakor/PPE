## Plotting results from dtlz1-6_interactive.py and dtlz7_interactive.py (3-objective DTLZ problemas). Adjust file paths and problem variants as needed. This is shown in Appendix B.3 of the paper.

import pickle
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
from pymoo.problems import get_problem
np.set_printoptions(precision=5)

#Initial Point:  [0.43292 0.21145 0.02439]
try:
    # Works in normal Python scripts
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for Jupyter/IPython
    FILE_DIR = Path(os.getcwd()).resolve()

z = 3 #7 # DTLZ number (1-6 for DTLZ1-6, 7 for DTLZ7)
n = 20 # 50# (50-dtlz7) #10 # (10-dtlz5, dtlz6) #20 # number of saved results
name = f"dtlz{z}"
problem = get_problem(name)
file_path = FILE_DIR.parent / f"toyEx_code/toy_results/ResultsMaths_dtlz{z}xqc"

file_pathD = FILE_DIR.parent / "toyEx_code/images/"

print("File path: ", file_pathD)



fig = plt.figure(figsize=(10,8),tight_layout = True)
ax = fig.add_subplot(111, projection='3d')
# Set up interactive mode
#plt.ion()
ax.scatter(problem.pareto_front()[:,0], problem.pareto_front()[:,1], problem.pareto_front()[:,2], color = "lightblue", s=30, alpha = 1,label = "True front",zorder=1)
with open(file_path / f'first_result_{n}.pkl', 'rb') as f:
    alphas, initial_point,predictor_point,corrector_point, preference = pickle.load(f)

    alphas, initial_point,predictor_point,corrector_point = np.array(alphas),np.array(initial_point),np.array(predictor_point),np.array(corrector_point)
    preference = np.array(preference)
    # Invert preferences for minimization visualization
    #preference[preference != 0] *= -1 

print(preference)
   
print("predictor_point shape: ", predictor_point[0, :])
print("Initial Point: ", initial_point)
for j in range(n):
    if j == 0 :
        #ax.plot(initial_point[:,0], initial_point[:,1], initial_point[:,2], label='Initial Point', color='r',linewidth=2)
        #ax.scatter(initial_point[0], initial_point[1], initial_point[2], label='Initial Point', color='r',s=50)
        ax.plot(initial_point[0], initial_point[1], initial_point[2],"ro", label='Initial Point',zorder=5)
        ax.plot([initial_point[0], predictor_point[j, 0]], [initial_point[1], predictor_point[j,1]], [initial_point[2], predictor_point[j,2]],"-", label='predictor step', color='g',linewidth=2,zorder=5)
        ax.plot([predictor_point[j ,0], corrector_point[j, 0]], [predictor_point[j, 1], corrector_point[j,1]], [predictor_point[j,2], corrector_point[j,2]],"-o", label='corrector Step', color='b',linewidth=2,zorder=5)
    if j > 0:
        ax.plot([corrector_pointold[0], predictor_point[j][0]], [corrector_pointold[1],predictor_point[j][1] ], [corrector_pointold[2], predictor_point[j][2]],"-o",  color='g',linewidth=2,zorder=5)
        ax.plot([predictor_point[j,0], corrector_point[j,0]], [predictor_point[j,1], corrector_point[j,1]], [predictor_point[j,2], corrector_point[j,2]],"-o", color='b',linewidth=2, zorder=5)

    print("Pref ", j, ": ", preference[j])
    print("Pred ", j, ": ", predictor_point[j])
    print("Corr ", j, ": ", corrector_point[j])

    corrector_pointold= corrector_point[j,:]

ax.set_xlabel(r"$f_1$",labelpad=10, fontsize = 20)
ax.set_ylabel(r"$f_2$",labelpad=10, fontsize = 20)
ax.set_zlabel(r"$f_3$",labelpad=10, fontsize = 20)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='z', labelsize=14)
ax.view_init(elev=8, azim=37, roll=-2)
ax.legend(fontsize = 12, loc="best")
#ax.view_init(elev=30, azim=60)
#ax.legend(fontsize = 12, loc="best")
#ax_pf.set_title('Objective Space')
#plt.savefig(f"{file_pathD}\\pareto_frontxcalldxcqs_{n}.png", dpi=300)
#plt.savefig(f"{file_pathD}\\pareto_frontxcalldtlz_{z}.png", dpi=300)
ax.grid(True)
plt.show()





# Example data
iterations = list(range(1, n + 1))  # Iteration numbers
f1 = predictor_point[:,0].tolist()  # Example f1 values
f2 = predictor_point[:,1].tolist()  # Example f2 values
f3 = predictor_point[:,2].tolist()  # Example f3 

f1_corr = corrector_point[:,0].tolist()  # Example f1 values
f2_corr = corrector_point[:,1].tolist()  # Example f2 values
f3_corr = corrector_point[:,2].tolist()  # Example f3 values



# weights corresponding to each fi
w1 = preference[:,0].tolist()  # Example weights for f1
w2 = preference[:,1].tolist()  # Example weights for f2
w3 = preference[:,2].tolist()  # Example weights for f3


colors = ['r', 'g', 'b']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
# --- Upper plot: f_i evolution ---
ax1.plot(iterations, f1, '-o', color=colors[0], label=r'$f_1$_pred', linewidth=0.9)
ax1.plot(iterations, f2, '-o', color=colors[1], label=r'$f_2$_pred',linewidth=0.9)
ax1.plot(iterations, f3, '-o', color=colors[2], label=r'$f_3$_pred',linewidth=0.9)

ax1.plot(iterations, f1_corr, '--x', color=colors[0], label=r'$f_1$_corr',linewidth=0.9)
ax1.plot(iterations, f2_corr, '--x', color=colors[1], label=r'$f_2$_corr',linewidth=0.9)
ax1.plot(iterations, f3_corr, '--x', color=colors[2], label=r'$f_3$_corr',linewidth=0.9)

ax1.set_ylabel(r'$objectives$', fontsize=14)
ax1.legend()
#ax1.set_title(f"Preference Objective {name}",   fontsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)


# --- Lower plot: weights evolution ---
#ax2.plot(iterations, w1, '-', color=colors[0], label=r'$\alpha_1$')
#ax2.plot(iterations, w2, '-', color=colors[1], label=r'$\alpha_2$')
#ax2.plot(iterations, w3, '-', color=colors[2], label=r'$\alpha_3$')
ax2.plot(iterations, w1, '-', color=colors[0], label=r'$\pi_1$')
ax2.plot(iterations, w2, '-', color=colors[1], label=r'$\pi_2$')
ax2.plot(iterations, w3, '-', color=colors[2], label=r'$\pi_3$')
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))


ax2.set_xlabel(r'Step-$n$', fontsize=14)
ax2.set_ylabel('Preferences', fontsize=14)

# --- Styling ---
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
#plt.savefig(f"{file_pathD}\\prefencedtlz_{z}.png", dpi=300)
plt.tight_layout()
plt.show()















