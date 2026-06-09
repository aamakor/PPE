## Plotting script for the results of the toy example. Adjust file paths and problem variants as needed. This is shown in Appendix B.3 of the paper. # (plotting results from 4MoreObj_dtlz_interactive.py)
# together with the radar plot of the corrector points and alpha values. The radar plot is shown in Appendix Figure 13 of the paper.
import pickle
#import matplotlib
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
np.set_printoptions(precision=5)

#Initial Point:  [0.43292 0.21145 0.02439]
try:
    # Works in normal Python scripts
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for Jupyter/IPython
    FILE_DIR = Path(os.getcwd()).resolve()

#file_path1 = FILE_DIR.parent / "toyEx_code/toy_results/ResultsMaths4dtlz2"
file_path1 = FILE_DIR.parent / "toyEx_code/toy_results/ResultsMaths5dtlz3"
file_pathD = FILE_DIR.parent / "toyEx_code/images"


n = 20 # Number of saved results


with open(file_path1 / f'first_result_20.pkl', 'rb') as f:
    alphas, initial_point,predictor_point,corrector_point, preference = pickle.load(f)

    alphas, initial_point,predictor_point,corrector_point = np.array(alphas),np.array(initial_point),np.array(predictor_point),np.array(corrector_point)
    preference = np.array(preference)
    # Invert preferences for minimization visualization
    #preference[preference != 0] *= -1 
   
#print("predictor_point shape: ", predictor_point[0, :])
#print("Initial Point: ", initial_point)


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 5),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

m = predictor_point.shape[1]
iterations = np.arange(1, predictor_point.shape[0] + 1)
colors = plt.cm.Set1.colors   #plt.cm.tab10.colors

for j in range(m):
    ax1.plot(iterations, predictor_point[:, j], '-o',
             color=colors[j % len(colors)],
             linewidth=0.9, alpha =1,label=fr'$f_{j+1}$_pred')

    ax1.plot(iterations, corrector_point[:, j], '--x',
             color=colors[j % len(colors)],
             linewidth=0.9, alpha =1, label=fr'$f_{j+1}$_corr')

    ax2.plot(iterations, preference[:, j],
             color=colors[j % len(colors)],alpha =1,
             label=fr'$\pi_{j+1}$')

ax1.set_ylabel('Objectives', fontsize=14)
ax2.set_ylabel("Preference", fontsize=14)
ax2.set_xlabel(r'Step-$n$', fontsize=14)  #Iteration
 
ax1.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.legend(ncol=1, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

ax1.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)

ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

#ax1.legend()
#ax1.set_title(f"Preference 4 Objectives (DTLZ2)",   fontsize=14)
#ax1.set_title(f"Preference 5 Objectives (DTLZ3)",   fontsize=14)



# --- Styling ---
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
#plt.savefig(f"{file_pathD}\\prefencexcqs_{n}.png", dpi=300)
#plt.savefig(f"{file_pathD}\\prefence4xc_{n}.png", dpi=300)
plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D



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



def example_data_corrector_and_alpha(corrector_points, preferences_alpha):
    n_steps, m_obj = corrector_points.shape

    #n_steps = 10

    data = []
    for i in range(n_steps):
        data.append([
            corrector_points[i].tolist(),
            preferences_alpha[i].tolist()
        ])

    return {
        'column names': [rf'$f_{i+1}$' for i in range(m_obj)],
        'group names': [f'Step {i+1}' for i in range(n_steps)],
        'data': data
    }





if __name__ == '__main__':

    # ---- fake example data (replace with real) ----
    #n_steps = 10
    #m_obj = 5
    n_steps, m_obj = corrector_point.shape

    # ---- get your data ----
    data_dict = example_data_corrector_and_alpha(corrector_point, alphas)

    labels = data_dict['column names']
    titles = data_dict['group names']
    radar_data = data_dict['data']

    N = len(labels)
    theta = radar_factory(N, frame='circle')

    # ---- subplot layout (auto) ----
    n_plots = len(radar_data)
    ncols = 4 #5
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
    labels_legend = ['optimal point', r'$\alpha^*$']

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

    #fig.suptitle('Preference 5 Objectives (DTLZ3)', size=14, y=1.03)
    plt.tight_layout()
    plt.show()











