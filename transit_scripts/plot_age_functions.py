import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

results = mne.externals.h5io.read_hdf5(
    'scores_univariate.h5'
)

sns.set_context(context='paper', font_scale=2)
sns.set_style('ticks', {'font.sans-serif': ['Helvetica'], 'pdf.fonttype': 42})
mpl.rcParams.update({'font.weight': '100'})

columns = list()
r2_scores = list()
big_scatter_map = list()
scatter_map = list()
for score, result in results.items():
    if not isinstance(result, tuple):
        result = (result,)
    for sub_score in result:
        name = sub_score['name']
        r2_bs = np.array([
            r2_score(x, xs) for xs, x, age_bs in sub_score['xs_bs']
        ])
        r2_scores.append(r2_bs)
        big_scatter_map.append(
            sub_score['xs_bs']
        )
        columns.append(name)

column_map = {
     'familiar_faces_details': 'Memory (familiar faces)',
     'Acc': 'Emotion Recognition (Accuracy)',
     'MinRT': 'Emotion Recognition (RT)',
     'TotalScore_y': 'Fluid Intelligence',
     'ToT_ratio': 'Memory (Tip of Tounge)',
     'Score': 'Proverbs',
     'Time': 'Executive Function (Hotel)',
     'PriPr': 'Memory Emotional 1',
     'ValPr': 'Memory Emotional 2',
     'ObjPr': 'Memory Emotional 3',
     'reactivity_neg_emo': 'Emotional Reactivity (negative)',
     'reactivity_pos_emo': 'Emotional Reactivity (positive)',
     'regulation_neg_emo': 'Emotion Regulation',
     'bp_dia_mean': 'Blood pressure (diastolic)',
     'TotalScore': 'Face Recognition',
     'force_match_direct': 'Force Matching (direct)',
     'force_match_indirect': 'Force Matching (indirect)',
     'trajectory_error': 'Motor Learning (error)',
     'movement_time': 'Motor Learning (time)'
 }


def color_boxes(bplot, colors, lw=3, transparent_faces=True):
    for k, elems in bplot.items():
        for jj, el in enumerate(elems):
            if len(elems) == len(colors):
                this_colors = colors[:]
            else:
                this_colors = sum([[c, c] for c in colors], [])
            if hasattr(el, 'set_color'):
                el.set_color(this_colors[jj])
            if hasattr(el, 'set_linewidth'):
                el.set_linewidth(lw)
    if transparent_faces:
        for box, cc in zip(bplot['boxes'], colors):
            if hasattr(box, 'set_facecolor'):
                box.set_facecolor(np.array(cc) * (1, 1, 1, 0.5))


def add_jittered_outliers(bpl, colors, alpha=0.15, width=0.4):
    for flier, cc in zip(bpl['fliers'], colors):
        flier.set_markerfacecolor(cc)
        flier.set(marker='.', alpha=0.15)
        ypos = flier.get_xydata()[:, 1]
        ypos += (np.random.random_sample((ypos.shape))
                 * width - (width / 2.))
        flier.set_ydata(ypos)


names = [column_map[kk] for kk in columns]
plt.figure(figsize=(8, 12))
sort_index = np.argsort(list(map(np.mean, r2_scores)))[::-1]
colors = mpl.cm.viridis(np.linspace(0.1, 0.9, len(r2_scores)))[::-1]
bpl = plt.boxplot(
    [r2_scores[ii] for ii in sort_index],
    labels=[names[ii] for ii in sort_index], whis=(2.5, 97.5),
    widths=[0.8] * len(r2_scores),
    vert=False, patch_artist=True)
color_boxes(bpl, colors)
add_jittered_outliers(bpl, colors)
plt.axvline(0, linestyle='--', color='black')
plt.xlabel('Explained Variance of Age')
sns.despine(trim=True)
