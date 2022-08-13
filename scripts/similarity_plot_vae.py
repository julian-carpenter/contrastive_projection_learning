import os
import numba
import matplotlib
import numpy as np
from sklearn import metrics
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


@numba.njit()
def w2_multi_normal(us1, us2):
    """
    This computes W2(p||q)
    See: https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions

    :param us1: stacked mean and logvariance vector of p
    :param us2: stacked mean and logvariance vector of q
    :return: W2 distance of p and q
    """

    # Covariance Matrix 1
    # We assume that the the mean and the variances are uncorrelated.
    # Then, the variances are the eigenvalues of the covariance matrix
    ds1 = int(len(us1) / 2)
    u1 = us1[:ds1]
    u2 = us2[:ds1]

    s1 = np.exp(us1[ds1:])
    s2 = np.exp(us2[ds1:])

    squaredcovs1 = np.sqrt(np.diag(s1))
    covs2 = np.diag(s2)
    trs1 = s1.sum()
    trs2 = s2.sum()
    squaredl2norm = np.power(np.linalg.norm(u2 - u1), 2)
    # The longer trace term 2 tr((s1^.5 s2 s1^.5)^.5)
    tr_covs = 2 * np.trace(
        np.sqrt(
            squaredcovs1.dot(covs2.dot(squaredcovs1))
        )
    )

    return squaredl2norm + trs1 + trs2 - tr_covs


matplotlib.use("agg")

seed = 5635786
base_path = os.path.join(os.path.curdir, "data")
class_names = [
    "Spherical/Oblate", "Round", "Elliptical", "Prolate", "Streak", "Bent",
    "Asymmetric", "NewtonRings", "DoubleRings", "Layered", "Empty"
]
#########################################################
# VAE Stuff
#########################################################
clr_file = os.path.join(base_path, "clr_polcart_0.1.npz")
with np.load(clr_file, allow_pickle=True) as f:
    # images have higher res in the clr file, they
    # have the same order as the vae file
    clr_images = f["images"]
vae_file = os.path.join(base_path, "VAE_vanilla_Burgess.npz")
with np.load(vae_file, allow_pickle=True) as fnpz:
    vae_mus = np.vstack(fnpz["mus"])
    vae_logvars = np.vstack(fnpz["logvars"])
    vae_labels = np.vstack(fnpz["labels"])

vae_projections = np.concatenate([vae_mus, vae_logvars], -1)

clr_images = np.array([(x - x.min()) / (x.max() - x.min()) for x in clr_images])
w2_vae_distances = metrics.pairwise_distances(vae_projections, None,
                                              w2_multi_normal)

# Here we choose a class combination that we seek similar images from.
# Get a list of all combinations that appear in the labels array.
# For the 11 class labels, there are 35 combinations
# Dispose those combinations that have the 'exclude_num' least images
# These are sometimes class combinations that have fewer than 5 images
exclude_num = 14
combination, count = np.unique(vae_labels, return_counts=True, axis=0)
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
num_similar_images = 6
# For the similarity plot we choose the following class combinations:
# 1: Spherical/Oblate, Elliptical, Asymmetric occurs 229 times
# 10: Spherical/Oblate, Elliptical, DoubleRings occurs 13 times
# 8: Prolate, Streak, Bent, Layered occurs 20 times
# 2: Prolate, Streak, Bent occurs 126 times
selected_class_combinations_idx = [1, 10, 8, 2]
selected_class_combinations = combination[selected_class_combinations_idx]
combination_bool_idx_vae = []
combination_real_idx_vae = []
for y in selected_class_combinations:
    bool_idx_tmp = []
    real_idx_tmp = []
    for i, x in enumerate(vae_labels):
        if (x == y).all():
            bool_idx_tmp.append(True)
            real_idx_tmp.append(i)
        else:
            bool_idx_tmp.append(False)
    combination_bool_idx_vae.append(bool_idx_tmp)
    combination_real_idx_vae.append(real_idx_tmp)
combination_bool_idx_vae = np.array(combination_bool_idx_vae)
# Select random images
rng = np.random.default_rng(seed=seed)
rand_idx = rng.integers(low=[0] * len(selected_class_combinations_idx),
                        high=combination_bool_idx_vae.sum(1),
                        size=[1, len(selected_class_combinations_idx)],
                        endpoint=False).T.squeeze()

selected_vae_distances = []
for ii, ri in enumerate(zip(rand_idx)):
    # Distances
    selected_vae_distances.append(w2_vae_distances[np.array(combination_real_idx_vae[ii])[ri]])
    # Label
    curr_vae_label = vae_labels[np.array(combination_real_idx_vae[ii])[ri]]
    # Sanity check
    assert (curr_vae_label == selected_class_combinations[ii]).all()

selected_vae_distances = np.array(selected_vae_distances)

# Now get the num_similar_images similar images
closest_vae_idx = [np.argsort(x.squeeze())[1:1 + num_similar_images] for x in selected_vae_distances]

######################################
# Plotting
######################################

nrows = len(selected_class_combinations_idx)
ncols = 1 + num_similar_images
enlarge_factor = 5
ratio_header = .1
plt.close("all")
context = plt.rc_context({"font.size": 38,
                          "text.usetex": True,
                          "text.latex.preamble": r"\usepackage{amsmath,nicefrac,cmbright,bm}"})
with context:
    f = plt.figure(constrained_layout=False,
                   figsize=(.8 * enlarge_factor * ncols * (1 + ratio_header), enlarge_factor * nrows)
                   )
    gs = GridSpec(nrows=nrows + 1,
                  ncols=ncols,
                  figure=f,
                  height_ratios=[ratio_header] + [1] * nrows)
    header_ax = f.add_subplot(gs[0, :])
    ax = []
    for x, y in product(range(1, nrows + 1), range(ncols)):
        ax.append(f.add_subplot(gs[x, y]))

    for ii, (ti, civae) in enumerate(zip(rand_idx, closest_vae_idx)):
        _idx = np.hstack([ti, civae])
        for iii, (a, iidx) in enumerate(zip(ax[ii * ncols:(ii + 1) * ncols], _idx)):
            if iii % ncols == 0:
                true_class_label = vae_labels[np.array(combination_real_idx_vae[ii])[iidx]].astype(int)
                add_str = None
                class_name = np.array(class_names)[true_class_label == 1]
                a.imshow(clr_images[np.array(combination_real_idx_vae[ii])[iidx]])
            else:
                class_label = vae_labels[iidx].astype(int)
                if ti == 10 and iidx == 757:
                    # We found an error in the ground truth due to this method :-)
                    # here we correct the ground truth
                    class_label[3] = 1
                    class_label[4] = 1
                score = (true_class_label * class_label).sum()
                score /= max(true_class_label.sum(), class_label.sum())
                # score = metrics.f1_score(true_class_label, class_label)
                add_str = "Overlap:\n"
                # add_str = "F1 score:\n"
                add_str += r"{{\bf {} }}".format("{:02.02f}".format(
                    score) if score > 0. else "0")
                # add_str += "\n"
                # add_str += r"{{\bf {} }} eV".format("{:02.02f}".format(
                #     vae_photon_energy[iidx]))
                class_name = np.array(class_names)[class_label == 1]
                a.imshow(clr_images[iidx])
            class_name_label = "\n".join([r"{{\bf {} }}".format(x) for x in class_name])
            a.text(.05, .95, class_name_label, c="w", fontsize=26,
                   transform=a.transAxes, va="top",
                   bbox=dict(facecolor="k", edgecolor="k", alpha=.25))
            if add_str is not None:
                a.text(.95, .05, add_str, c="w", fontsize=26,
                       transform=a.transAxes, va="bottom", ha="right",
                       bbox=dict(facecolor="k", edgecolor="k", alpha=.25))
            a.axis(False)

    header_ax.axis(False)
    r_str = r"{{\bf i)}} The most similar images using both frameworks"
    header_ax.text(0.025, 1,
                   s=r_str,
                   c="w",
                   ha="left",
                   va="bottom",
                   transform=header_ax.transAxes)

    ax_width = header_ax.get_position().width
    text_pos = ax_width / ncols / 2
    r_str = r"Input image"
    header_ax.text(text_pos * 1.15, 0,
                   s=r_str,
                   c="k",
                   ha="center",
                   va="top",

                   transform=header_ax.transAxes)

    # Create a Rectangle patch, positions have to be tuned manually
    ver_bar_1 = patches.Rectangle(
        (.1465, .05),
        .005, .8,
        linewidth=0,
        facecolor=".875",
        zorder=-1,
        transform=f.transFigure)
    hor_bar_1 = patches.Rectangle(
        (0.05, .228),
        .9, .005 * 3.1 / 2,
        linewidth=0,
        facecolor=".875",
        zorder=-1,
        transform=f.transFigure)
    hor_bar_2 = patches.Rectangle(
        (0.05, .228 * 2),
        .9, .005 * 3.1 / 2,
        linewidth=0,
        facecolor=".875",
        zorder=-1,
        transform=f.transFigure)
    hor_bar_3 = patches.Rectangle(
        (0.05, .228 * 3),
        .9, .005 * 3.1 / 2,
        linewidth=0,
        facecolor=".875",
        zorder=-1,
        transform=f.transFigure)
    f.patches.extend([ver_bar_1, hor_bar_1, hor_bar_2, hor_bar_3])

    r_str = r"{{\bf i)}} The most similar images using the contrastive projection learning framework"
    f.text(.015, .945,
           s=r_str,
           c="k",
           ha="left",
           va="bottom",
           fontsize=46,
           transform=f.transFigure)

    text_pos = ax_width
    r_str = r"Closest images in the VAE latent space"
    header_ax.text(text_pos * .7, 0,
                   s=r_str,
                   c="k",
                   ha="center",
                   va="top",
                   transform=header_ax.transAxes)

    f.tight_layout()
    ff = os.path.join(os.path.curdir,
                      "similarity_vae.pdf")
    f.savefig(ff, dpi=300)
    str_1 = "pdfcrop --margins 5 {} {}".format(ff, ff)
    os.system(str_1)
