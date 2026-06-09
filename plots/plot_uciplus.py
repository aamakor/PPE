# This file contains code for loading and visualizing the results of the Preference Pareto Exploration (PPE)
#  framework on the UCI Adult dataset with 5 objectives (age, education, marital status, race, and sex) 
# and comparing the training and test trajectories of the predictor and corrector points along
# with the preferences across the steps. We also visualize the optimal points obtained across different 
# runs in a radar plot for both training and test sets. 
# This corresponds to Section 4.1  Figure 6 of the paper and  Figure 14, and Figure 15 in the Appendix.    


import pickle
import matplotlib
matplotlib.use('webAgg')
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
np.set_printoptions(precision=5)
import math
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def is_not_dominated(K, b):
    """
    # Function to identify suboptimal points that are not on the current front
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

file_path = FILE_DIR.parent / "Results/UCI_plus"

n = 20 # Number of saved results

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



final_preference = []
final_opt_pred = []
final_opt_corr = []


all_subopt_points = np.concatenate((corr_all,initial_point[-1, :].reshape(1,5)),axis=0)

for i in range(len(all_subopt_points)):
    b = all_subopt_points[i]
    if is_not_dominated(all_subopt_points, b):
        idx = np.where((corr_all == b).all(axis=1))[0]
        if len(idx) > 0:
            final_preference.append(pref_all[idx[0]])
            final_opt_pred.append(pred_all[idx[0]])
            final_opt_corr.append(corr_all[idx[0]])

final_preferencet = []
final_opt_predt = []
final_opt_corrt = []

all_subopttest_points = np.concatenate((corrt_all,initialt_point[-1, :].reshape(1,5)),axis=0)

for i in range(len(all_subopttest_points)):
    bt = all_subopttest_points[i]
    if is_not_dominated(all_subopttest_points, bt):
        idx = np.where((corrt_all == bt).all(axis=1))[0]
        if len(idx) > 0:
            final_preferencet.append(preft_all[idx[0]])
            final_opt_predt.append(predt_all[idx[0]])
            final_opt_corrt.append(corrt_all[idx[0]])

    
predictor_point = np.array(final_opt_pred)   
corrector_point = np.array(final_opt_corr)   
predictor_pointt = np.array(final_opt_predt)
corrector_pointt = np.array(final_opt_corrt)
preference = np.array(final_preference)
preferencet = np.array(final_preferencet)


## Scaling for better visualization as shown in figure 5a of the paper using z-score normalization (standardization) for both training and test points. 
# This will help in visualizing the points more clearly in the 2D plot. We remove the initial point from the scaling and only scale the subsequent points for better visualization of the trajectory.
predictor_points = (predictor_point - predictor_point.mean(axis=0)) / predictor_point.std(axis=0) 
corrector_points = (corrector_point - corrector_point.mean(axis=0))  / corrector_point.std(axis=0) 

predictor_pointts = (predictor_pointt - predictor_pointt.mean(axis=0)) / predictor_pointt.std(axis=0) 
corrector_pointts = (corrector_pointt - corrector_pointt.mean(axis=0))  / corrector_pointt.std(axis=0) 

############## Plotting Training set ########################

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 5),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

m = predictor_points.shape[1]
iterations = np.arange(1, predictor_points.shape[0] + 1)
colors = plt.cm.Set1.colors   #plt.cm.tab10.colors

label = ["age", "edu", "ms",   "race", "sex"]

for j in range(m):
    ax1.plot(iterations, predictor_points[:, j], '-o',
             color=colors[j % len(colors)],
             linewidth=0.9, alpha =1,label=f'{label[j]}_pred')
    if j==0:
        print(predictor_points[0, :])

    ax1.plot(iterations, corrector_points[:, j], '--x',
             color=colors[j % len(colors)],
             linewidth=0.9, alpha =1, label=f'{label[j]}_corr')

    ax2.plot(iterations, preference[:, j],
             color=colors[j % len(colors)],alpha =1,
             label=fr'$\pi_{{{label[j]}}}$')

ax1.set_ylabel(r'$Objectives$', fontsize=14)
ax2.set_ylabel("Preference", fontsize=14)
ax2.set_xlabel(r"Step-$n$", fontsize=14)
ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
ax1.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

# --- Styling ---
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



############## Plotting Test set ########################

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 5),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

m = predictor_pointts.shape[1]
iterations = np.arange(1, predictor_pointts.shape[0] + 1)
colors = plt.cm.Set1.colors   #plt.cm.tab10.colors

label = ["age", "edu", "ms",  "race", "sex"]

for j in range(m):
    ax1.plot(iterations, predictor_pointts[:, j], '-o',
             color=colors[j % len(colors)],
             linewidth=0.9, alpha =1,label=f'{label[j]}_pred')
    
    if j==0:
        print(predictor_pointts[0, :])

    ax1.plot(iterations, corrector_pointts[:, j], '--x',
             color=colors[j % len(colors)],
             linewidth=0.9, alpha =1, label=f'{label[j]}_corr')

    ax2.plot(iterations, preferencet[:, j],
             color=colors[j % len(colors)],alpha =1,
             label=fr'$\pi_{{{label[j]}}}$')

ax1.set_ylabel(r'$objectives$', fontsize=14)
ax2.set_ylabel("Preference", fontsize=14)
ax2.set_xlabel(r'Step-$n$', fontsize=14)

ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

ax1.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)

ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

# --- Styling ---
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



##### Radar plot for training and test set (Figure 14 and Figure 15 of the Appendix) #####



def radar_factory(num_vars, frame='polygon'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return super()._gen_axes_patch()
            return plt.Polygon(self._unit_poly_verts(theta), closed=True)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            spine = Spine(axes=self,
                          spine_type='circle',
                          path=Path(self._unit_poly_verts(theta)))
            spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                 + self.transAxes)
            return {'polar': spine}

        def _unit_poly_verts(self, theta):
            x0, y0, r = [0.5]*3
            return [(r*np.cos(t)+x0, r*np.sin(t)+y0) for t in theta]

    register_projection(RadarAxes)
    return theta



def example_data_corrector_and_pref(corrector_points, preferences):
    n_steps, m_obj = corrector_points.shape

    #n_steps = 10

    data = []
    for i in range(n_steps):
        data.append([
            corrector_points[i].tolist(),
            preferences[i].tolist()
        ])

    return {
        'column names': ["age", "edu", "ms",   "race", "sex"],
        'group names': [f'Step {i+1}' for i in range(n_steps)],
        'data': data
    }





if __name__ == '__main__':

    # ---- fake example data (replace with real) ----
    #n_steps = 10
    #m_obj = 5
    n_steps, m_obj = corrector_point.shape

    # ---- get your data ----
    data_dict = example_data_corrector_and_pref(corrector_point, alpha_all)

    data_dicttest = example_data_corrector_and_pref(corrector_pointt, alpha_all)

    labels = data_dict['column names']
    titles = data_dict['group names']
    radar_data = data_dict['data']

    N = len(labels)
    theta = radar_factory(N, frame='circle')

    # ---- subplot layout (auto) ----
    n_plots = len(radar_data)
    ncols = 4
    nrows = math.ceil(n_plots / ncols)

    fig, axs = plt.subplots(
        figsize=(4*ncols, 5*nrows),
        nrows=nrows,
        ncols=ncols,
        subplot_kw=dict(projection='radar')
    )

    axs = np.atleast_1d(axs).flatten()

    # ---- plot ----
    colors = ['tab:blue', 'tab:red']
    labels_legend = ['optimal points', fr'$\alpha^*$']

    for ax, title, case_data in zip(axs, titles, radar_data):
        ax.set_title(title, size=14, pad=12)
        ax.set_varlabels(labels)
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
            #label.set_fontweight('bold')

        for d, color, lbl in zip(case_data, colors, labels_legend):
            ax.plot(theta, d, color=color, linewidth=2, label=lbl)
            ax.fill(theta, d, color=color, alpha=0.25)
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    #ax.set_varlabels(labels)
    # hide unused axes
    for ax in axs[len(radar_data):]:
        ax.set_visible(False)
    
    

    #fig.suptitle("Preference 5 Objectives (UCI)-Train", size=14, y=1.03)
    plt.tight_layout()
    plt.show()


    fig, axs = plt.subplots(
        figsize=(5*ncols, 5*nrows),
        nrows=nrows,
        ncols=ncols,
        subplot_kw=dict(projection='radar')
    )

    axs = np.atleast_1d(axs).flatten()

    # ---- plot ----
    colors = ['tab:blue', 'tab:red']
    labels_legend = ['optimal points', fr'$\alpha^*$']

    for ax, title, case_data in zip(axs, titles, data_dicttest['data']):
        ax.set_title(title, size=14, pad=12)
        ax.set_varlabels(labels)
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
            #label.set_fontweight('bold')

        for d, color, lbl in zip(case_data, colors, labels_legend):
            ax.plot(theta, d, color=color, linewidth=2, label=lbl)
            ax.fill(theta, d, color=color, alpha=0.25)
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    #ax.set_varlabels(labels)
    # hide unused axes
    for ax in axs[len(radar_data):]:
        ax.set_visible(False)

    #fig.suptitle("Preference 5 Objectives (UCI)-Test", size=14, y=1.03)
    plt.tight_layout()
    plt.show()

