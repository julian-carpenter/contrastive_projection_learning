import os
import matplotlib
import numpy as np
from sklearn import metrics, preprocessing
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

sns.set_style("white")
matplotlib.use("agg")

seed = 5635786
base_path = os.path.join(os.path.curdir, "data")
class_names = [
    "Spherical/Oblate", "Round", "Elliptical", "Prolate", "Streak", "Bent",
    "Asymmetric", "NewtonRings", "DoubleRings", "Layered", "Empty"
]
#########################################################
# CLR Stuff
#########################################################
clr_file = os.path.join(base_path, "clr_polcart_0.1.npz")
with np.load(clr_file, allow_pickle=True) as f:
    clr_images = f["images"]
    clr_projections = f["projections"]
    clr_labels = f["labels"]
    clr_radius = f["radii"]
    clr_photon_energy = f["photon_energies"]

clr_images = np.array([(x - x.min()) / (x.max() - x.min()) for x in clr_images])
cosine_clr_distances = metrics.pairwise_distances(clr_projections, None,
                                                  "cosine")

# Here we choose a class combination that we seek similar images from.
# Get a list of all combinations that appear in the labels array.
# For the 11 class labels, there are 35 combinations
# Dispose those combinations that have the 'exclude_num' least images
# These are sometimes class combinations that have fewer than 5 images
exclude_num = 16
combination, count = np.unique(clr_labels, return_counts=True, axis=0)
sort_perm = count.argsort()
combination, count = combination[sort_perm[::-1]], count[sort_perm[::-1]]
combination, count = combination[:-exclude_num], count[:-exclude_num]
# manual removing two entries without "oblate" or "prolate" superordinate class
combination = np.vstack([combination[:13], combination[14:-1]])
count = np.hstack([count[:13], count[14:-1]])

# Print out the final set of combinations from which we choose
names = []
for c, co in zip(combination, count):
    n = ", ".join(list(np.array(class_names)[c == 1]))
    names.append(n)

for i, (n, c) in enumerate(zip(names, count)):
    print("{}: {} occurs {} times".format(i, n, c))

# Take the three most similar images from each embedding distances
num_similar_images = 4
# For the similarity plot we choose the following class combinations:
# 1: Spherical/Oblate, Elliptical, Asymmetric occurs 229 times
# 10: Spherical/Oblate, Elliptical, DoubleRings occurs 13 times
# 8: Prolate, Streak, Bent, Layered occurs 20 times
# 2: Prolate, Streak, Bent occurs 126 times
selected_class_combinations_idx = [1, 10, 8, 2]
# selected_class_combinations_idx = [1, 10, 8, 2]
selected_class_combinations = combination[selected_class_combinations_idx]
combination_bool_idx_clr = []
combination_real_idx_clr = []
for y in selected_class_combinations:
    bool_idx_tmp = []
    real_idx_tmp = []
    for i, x in enumerate(clr_labels):
        if (x == y).all():
            bool_idx_tmp.append(True)
            real_idx_tmp.append(i)
        else:
            bool_idx_tmp.append(False)
    combination_bool_idx_clr.append(bool_idx_tmp)
    combination_real_idx_clr.append(real_idx_tmp)
combination_bool_idx_clr = np.array(combination_bool_idx_clr)
# Select random images
rng = np.random.default_rng(seed=seed)
rand_idx = rng.integers(low=[0] * len(selected_class_combinations_idx),
                        high=combination_bool_idx_clr.sum(1),
                        size=[1, len(selected_class_combinations_idx)],
                        endpoint=False).T.squeeze()
selected_class_combinations = selected_class_combinations[1:]
selected_class_combinations_idx = selected_class_combinations_idx[1:]
rand_idx = rand_idx[1:]
combination_bool_idx_clr = combination_bool_idx_clr[1:]
combination_real_idx_clr = combination_real_idx_clr[1:]
selected_clr_distances = []
for ii, ri in enumerate(zip(rand_idx)):
    # Distances
    selected_clr_distances.append(cosine_clr_distances[np.array(combination_real_idx_clr[ii])[ri]])
    # Label
    curr_clr_label = clr_labels[np.array(combination_real_idx_clr[ii])[ri]]
    # Sanity check
    assert (curr_clr_label == selected_class_combinations[ii]).all()

selected_clr_distances = np.array(selected_clr_distances)

# Now get the num_similar_images similar images
closest_clr_idx = [np.argsort(x.squeeze())[1:1 + num_similar_images] for x in selected_clr_distances]

######################################
# Plotting
######################################

nrows = len(selected_class_combinations_idx)
ncols = 1 + num_similar_images
enlarge_factor = 3
plt.close("all")
context = plt.rc_context({"font.size": 24,
                          "text.usetex": True,
                          "text.latex.preamble": r"\usepackage{amsmath,nicefrac,cmbright,bm}"})
with context:
    f = plt.figure(constrained_layout=False,
                   figsize=(enlarge_factor * ncols,
                            enlarge_factor * nrows)
                   )
    gs = GridSpec(nrows=nrows,
                  ncols=ncols,
                  figure=f,
                  )
    ax = []
    for x, y in product(range(0, nrows), range(ncols)):
        ax.append(f.add_subplot(gs[x, y]))

    for ii, (ti, ciclr) in enumerate(zip(rand_idx[::-1], closest_clr_idx[::-1])):
        _idx = np.hstack([ti, ciclr])
        for iii, (a, iidx) in enumerate(zip(ax[ii * ncols:(ii + 1) * ncols], _idx)):
            if iii % ncols == 0:
                true_class_label = clr_labels[np.array(combination_real_idx_clr[::-1][ii])[iidx]].astype(int)
                true_class_label[0] = 0  # Oblate always to zero
                true_class_label[3] = 0  # Prolate always to zero
                add_str = None
                class_name = np.array(class_names)[true_class_label == 1]
                a.imshow(clr_images[np.array(combination_real_idx_clr[::-1][ii])[iidx]])
            else:
                class_label = clr_labels[iidx].astype(int)
                if ti == 10 and iidx == 757:
                    # We found an error in the ground truth due to this method :-)
                    # here we correct the ground truth
                    class_label[3] = 1
                    class_label[4] = 1
                if ti == 10 and iidx == 1005:
                    class_label[2] = 1
                class_label[0] = 0  # Oblate always to zero
                class_label[3] = 0  # Prolate always to zero
                score = (true_class_label * class_label).sum()
                score /= max(true_class_label.sum(), class_label.sum())
                add_str = "Overlap:\n"
                add_str += r"{{\bf {} }}".format("{:02.02f}".format(
                    score) if score > 0. else "0")
                class_name = np.array(class_names)[class_label == 1]

                print(ti, iidx, class_name, add_str)
                a.imshow(clr_images[iidx])
            class_name_label = "\n".join([r"{{\bf {} }}".format(x) for x in class_name])
            a.text(.05, .95, class_name_label, c="w",
                   transform=a.transAxes, va="top",
                   )
            if add_str is not None:
                a.text(.95, .05, add_str, c="w",
                       transform=a.transAxes, va="bottom", ha="right",
                       )
            a.axis(False)

    f.tight_layout(pad=1e-3)
    ff = os.path.join(os.path.curdir,
                      "similarity_cplr.pdf")
    f.savefig(ff, dpi=300)
    str_1 = "pdfcrop --margins 5 {} {}".format(ff, ff)
    os.system(str_1)
