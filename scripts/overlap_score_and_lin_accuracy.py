import os
import numba
import umap
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model, pipeline, \
    preprocessing, decomposition, model_selection, manifold
from matplotlib.backends.backend_pdf import PdfPages


def save_to_pdf(path, dataframe, title):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(int(dataframe.columns.shape[0]),
                                    int(.35 * dataframe.index.shape[0])))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title)
    the_table = ax.table(cellText=dataframe.values,
                         colLabels=dataframe.columns,
                         rowLabels=dataframe.index,
                         loc='center', edges="open",
                         cellLoc="center", rowLoc="right")
    fig.tight_layout()
    pp = PdfPages(path)
    pp.savefig(fig, bbox_inches='tight', dpi=300)
    pp.close()


def overlap_score(true_lbl, other_lbl, rel_idx=None):
    """
    This computes a simple overlap score.
    :param true_lbl: The True label with which the overlap should be computed
    :param other_lbl: The other label that should overlap with true_lbl
    :param rel_idx: If only selected indices should count
    :return: Scalar -> The overlap score
    """

    # First do a vector multiplication
    if rel_idx is not None:
        true_lbl = true_lbl[rel_idx]
        other_lbl = other_lbl[rel_idx]
    score = (true_lbl * other_lbl).sum()
    # Then normalize by the max number of positive labels in either true_lbl oder other_lbl
    score /= max(true_lbl.sum(), other_lbl.sum())

    return score


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
    # We assume that the mean and the variances are uncorrelated.
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


def lin_acc(embeddings, labels):
    """
    This trains a linear classifier and returns the accuracy, precision,
    recall and all confusion matrix elements
    :param embeddings: The trained embeddings ... This is X
    :param labels:  The target ... this is Y
    :return: Multiple scalars
    """
    clf = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        # We define the SGDClassifier to be as close as possible to the
        # linear classifier in Chen, T., et al. A simple framework for
        # contrastive learning of visual representations. arXiv (2020).
        linear_model.SGDClassifier(loss="perceptron", eta0=1,
                                   learning_rate="invscaling",
                                   penalty="l2", alpha=1e-4,  # alpha with l2 is weight decay
                                   shuffle=True,  # shuffle for each epoch using the same seed
                                   random_state=seed)
    )
    scoring_dict = {
        "accuracy": metrics.make_scorer(metrics.accuracy_score),
        "f1": metrics.make_scorer(lambda x, y: metrics.f1_score(x, y, zero_division=0)),
        "precision": metrics.make_scorer(lambda x, y: metrics.precision_score(x, y, zero_division=0)),
        "recall": metrics.make_scorer(lambda x, y: metrics.recall_score(x, y, zero_division=0))
    }
    scores = model_selection.cross_validate(clf, embeddings, labels,
                                            cv=5, scoring=scoring_dict
                                            )
    acc = scores["test_accuracy"].mean()
    prec = scores["test_precision"].mean()
    rec = scores["test_recall"].mean()
    f1 = scores["test_f1"].mean()
    return acc, prec, rec, f1


def get_scores(embs, lbls, mode="clr"):
    """
    :param embs: The embeddings (n_samples, n_features)
    :param lbls: The labels (n_samples, n_labels)
    :param mode: String: Either 'clr', 'pca', or 'rand'
    :return: Two dictionaries with micro and macro averaves of all scores
    """

    # First, get the pairwise distances
    if mode.lower() == "vae":
        metric = w2_multi_normal
    elif mode.lower() == "clr":
        metric = "cosine"
    elif mode.lower() == "pca":
        metric = "euclidean"
    elif mode.lower() == "pcacorr":
        # Largest correlation should yield the smallest 'distance'
        metric = lambda x, y: 1 - np.abs(np.corrcoef(x, y))[0, 1]
    elif mode.lower() == "rand":
        # This is just for testing and currently identical to 'pca'
        metric = "euclidean"
    else:
        raise KeyError("Mode must be either 'vae,' 'clr', 'rand', or 'pca'")

    distances = metrics.pairwise_distances(X=embs,
                                           metric=metric)

    num_similar_images = np.round(0.01 * len(lbls)).astype(int)
    # Now get the overlap score for only nonzero and manually selected relevant classes
    class_label_count = lbls.sum(0)
    relevant_idx = [class_names.index(x) for x in relevant_classes]
    # Empty lists for storing the scores
    overlap_score_class_averages = []
    if mode == "vae":
        embedding = manifold.Isomap(n_components=int(embs.shape[-1] / 2),
                                    n_neighbors=num_similar_images,
                                    metric=w2_multi_normal)
        embs = embedding.fit_transform(embs)
    # we iterate over all classes
    for ii, (clc, cl) in enumerate(zip(class_label_count, class_names)):
        # proceed if not empty and relevant
        if clc > 0 and ii in relevant_idx:
            # get the indices of the current class
            indexes = np.argwhere(lbls[:, ii] == 1).squeeze()
            # get the indices of the images with the smallest distance for all images
            closest_idx = [np.argsort(x.squeeze())[1:1 + num_similar_images] for x in distances]
            # now get the closest images for the current class
            closest_idx = np.array(closest_idx, dtype=object)[indexes].squeeze()
            # calculate the overlap score
            overlap_micro_avg = []
            for iii, ll in enumerate(np.squeeze(lbls[indexes])):
                single_scores = []
                for y in np.squeeze(lbls[closest_idx[iii].astype(int)]):
                    single_scores.append(overlap_score(ll, y, relevant_idx))
                overlap_micro_avg.append(np.nanmean(single_scores))
            overlap_score_ = np.nanmean(overlap_micro_avg)

            # append to the lists
            overlap_score_class_averages.append(overlap_score_)

    # Convert micro averages to numpy and calculate macro averages
    overlap_score_class_averages = np.array(overlap_score_class_averages)
    global_overlap_score_class_average = np.nanmean(overlap_score_class_averages)

    # Linear classifier score
    # Placeholder lists
    avg_acc = []
    avg_prec = []
    avg_rec = []
    avg_f1 = []
    # iterate over all classes
    for ii, cl in enumerate(class_names):
        # get the current class
        target = lbls[:, ii]
        # if not zero and relevant
        if target.sum() > 0 and ii in relevant_idx:
            # train a linear classifier and get the metrics
            acc, prec, rec, f1 = lin_acc(embeddings=embs,
                                         labels=target)
            # append to lists
            avg_f1.append(f1)
            avg_acc.append(acc)
            avg_prec.append(prec)
            avg_rec.append(rec)

    # convert to numpy
    avg_f1 = np.array(avg_f1)
    avg_acc = np.array(avg_acc)
    avg_prec = np.array(avg_prec)
    avg_rec = np.array(avg_rec)

    # prepare the dictionaries; One for micro averages and one for macro averages
    avg_score_dict = {
        "sim_overlap": to_trunc_str([global_overlap_score_class_average])[0],
        "lin_class_acc": to_trunc_str([avg_acc.mean()])[0],
        "lin_class_f1": to_trunc_str([avg_f1.mean()])[0],
        "lin_class_prec": to_trunc_str([avg_prec.mean()])[0],
        "lin_class_rec": to_trunc_str([avg_rec.mean()])[0],
    }
    class_score_dict = {
        "sim_overlap": to_trunc_str(overlap_score_class_averages),
        "lin_class_acc": to_trunc_str(avg_acc),
        "lin_class_f1": to_trunc_str(avg_f1),
        "lin_class_prec": to_trunc_str(avg_prec),
        "lin_class_rec": to_trunc_str(avg_rec),
    }

    return avg_score_dict, class_score_dict


def to_trunc_str(ll):
    """
    Helper function to convert a float number to a nice string
    :param ll: float scalar
    :return: list with one string
    """
    return ["{:02.02f}".format(x) for x in ll]


# Base path
data_base_path = "/scratch/jzimmermann/contrastive_learning"
# data_base_path = os.path.join(os.path.curdir, "contrastive_projection_learning_models")
# Output path
data_out_path = "/scratch/jzimmermann/unsupervised_learning_data"
# data_out_path = os.path.join(data_base_path, "unsupervised_learning_data")

if not os.path.isdir(data_out_path):
    os.makedirs(data_out_path)
# the random seed to ensure reproducibility
seed = 5635786
# Produce random shuffle indices to equally shuffle the dataset
rng = np.random.default_rng(seed=seed)
rand_idx = np.arange(1260)
rng.shuffle(rand_idx)
# class names in the dataset
class_names = [
    "Spherical/Oblate",
    "Round",
    "Elliptical",
    "Prolate",
    "Streak",
    "Bent",
    "Asymmetric",
    "NewtonRings",
    "DoubleRings",
    "Layered",
    "Empty"
]
# the relevant classes for all scores
relevant_classes = ("Elliptical", "Streak", "Bent", "Asymmetric",
                    "NewtonRings", "DoubleRings", "Layered")

#########################################################
# CLR
#########################################################
# Get filenames of all npz files
file_names = glob(os.path.join(data_base_path, "*", "*.npz"))
# Iterate over the npz files
for i, f in enumerate(file_names):
    # first we isolate the dirname as a column label in subsequent tables
    dirname = os.path.basename(os.path.dirname(f))
    col_name = "{}".format(dirname.replace("_", " "))
    # Given a dirname of 'polcart_clr_temp_0.1' this yields 'clr polcart 0.1'
    col_name = "clr " + col_name[:col_name.find(" ")] + col_name[-col_name[::-1].find(" ") - 1:]
    print(col_name, flush=True)
    # Load the file
    with np.load(f, allow_pickle=True) as fnpz:
        clr_projections = fnpz["projections"][rand_idx]
        clr_labels = fnpz["labels"][rand_idx]
    # get micro and macro averages of all relevant scores
    clr_avg_score_dict, clr_class_score_dict = get_scores(clr_projections,
                                                          clr_labels,
                                                          mode="clr")
    ## MACRO AVERAGES DATAFRAME
    # if this is the first iteration build the initial pandas dataframe
    if i == 0:
        df = pd.DataFrame.from_dict(clr_avg_score_dict,
                                    orient="index",
                                    columns=[col_name]
                                    )
    else:  # else, append to existing pandas dataframe
        df2 = pd.DataFrame.from_dict(clr_avg_score_dict,
                                     orient="index",
                                     columns=[col_name]
                                     )
        df = df.join(df2)

    ## MICRO AVERAGES DATAFRAME
    # add an index title key to the dictionary
    # Pandas will use this as index labels
    clr_class_score_dict.update({"Index Title": relevant_classes})

    # Build the dataframe, write-out to pickle and save a pdf file
    df_class = pd.DataFrame.from_dict(clr_class_score_dict,
                                      orient="columns",
                                      ).set_index('Index Title')
    df_class.to_pickle(os.path.join(data_out_path,
                                    "micro_avg_scores_{}.pkl".format(dirname))
                       )
    save_to_pdf(os.path.join(data_out_path,
                             "micro_avg_scores_{}.pdf".format(dirname)),
                df_class,
                col_name)
#########################################################
# PCA
#########################################################
print("pca", flush=True)
with np.load(file_names[0], allow_pickle=True) as fnpz:
    clr_images = fnpz["images"][rand_idx]
    clr_labels = fnpz["labels"][rand_idx]
clr_imgs_flat = clr_images.reshape([clr_images.shape[0], -1])
clr_imgs_flat = preprocessing.MinMaxScaler().fit_transform(clr_imgs_flat)
pca_clr = decomposition.PCA(clr_projections.shape[-1],
                            random_state=seed).fit(clr_imgs_flat)
pca_embs = pca_clr.transform(clr_imgs_flat)
pca_avg_score_dict, pca_class_score_dict = get_scores(pca_embs,
                                                      clr_labels,
                                                      mode="pca")
df2 = pd.DataFrame.from_dict(pca_avg_score_dict,
                             orient="index",
                             columns=["pca"]
                             )
df = df.join(df2)

# Now for the individual classes
pca_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(pca_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_pca.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_pca.pdf"), df_class,
            "pca")
#########################################################
# PCA Correlation Metric
#########################################################
print("pcacorr", flush=True)
with np.load(file_names[0], allow_pickle=True) as fnpz:
    clr_images = fnpz["images"][rand_idx]
    clr_labels = fnpz["labels"][rand_idx]
clr_imgs_flat = clr_images.reshape([clr_images.shape[0], -1])
clr_imgs_flat = preprocessing.MinMaxScaler().fit_transform(clr_imgs_flat)
pcacorr_clr = decomposition.PCA(clr_projections.shape[-1],
                                random_state=seed).fit(clr_imgs_flat)
pcacorr_embs = pcacorr_clr.transform(clr_imgs_flat)
pcacorr_avg_score_dict, pcacorr_class_score_dict = get_scores(pcacorr_embs,
                                                              clr_labels,
                                                              mode="pcacorr")
df2 = pd.DataFrame.from_dict(pcacorr_avg_score_dict,
                             orient="index",
                             columns=["pcacorr"]
                             )
df = df.join(df2)

# Now for the individual classes
pcacorr_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(pcacorr_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_pcacorr.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_pcacorr.pdf"), df_class,
            "pcacorr")
#########################################################
# KPCA
#########################################################
print("kpca", flush=True)
with np.load(file_names[0], allow_pickle=True) as fnpz:
    clr_images = fnpz["images"][rand_idx]
    clr_labels = fnpz["labels"][rand_idx]
clr_imgs_flat = clr_images.reshape([clr_images.shape[0], -1])
clr_imgs_flat = preprocessing.MinMaxScaler().fit_transform(clr_imgs_flat)
kpca_clr = decomposition.KernelPCA(clr_projections.shape[-1],
                                   random_state=seed).fit(clr_imgs_flat)
kpca_embs = kpca_clr.transform(clr_imgs_flat)
kpca_avg_score_dict, kpca_class_score_dict = get_scores(kpca_embs,
                                                        clr_labels,
                                                        mode="pca")
df2 = pd.DataFrame.from_dict(kpca_avg_score_dict,
                             orient="index",
                             columns=["kpca"]
                             )
df = df.join(df2)

# Now for the individual classes
kpca_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(kpca_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_kpca.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_kpca.pdf"), df_class,
            "kpca")
#########################################################
# KPCA Correlation Metric
#########################################################
print("kpcacorr", flush=True)
with np.load(file_names[0], allow_pickle=True) as fnpz:
    clr_images = fnpz["images"][rand_idx]
    clr_labels = fnpz["labels"][rand_idx]
clr_imgs_flat = clr_images.reshape([clr_images.shape[0], -1])
clr_imgs_flat = preprocessing.MinMaxScaler().fit_transform(clr_imgs_flat)
kpcacorr_clr = decomposition.KernelPCA(clr_projections.shape[-1],
                                       random_state=seed).fit(clr_imgs_flat)
kpcacorr_embs = kpcacorr_clr.transform(clr_imgs_flat)
kpcacorr_avg_score_dict, kpcacorr_class_score_dict = get_scores(kpcacorr_embs,
                                                                clr_labels,
                                                                mode="pcacorr")
df2 = pd.DataFrame.from_dict(kpcacorr_avg_score_dict,
                             orient="index",
                             columns=["kpcacorr"]
                             )
df = df.join(df2)

# Now for the individual classes
kpcacorr_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(kpcacorr_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_kpcacorr.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_kpcacorr.pdf"), df_class,
            "kpcacorr")
#########################################################
# Factor Analysis
#########################################################
print("fa", flush=True)
with np.load(file_names[0], allow_pickle=True) as fnpz:
    clr_images = fnpz["images"][rand_idx]
    clr_labels = fnpz["labels"][rand_idx]
clr_imgs_flat = clr_images.reshape([clr_images.shape[0], -1])
clr_imgs_flat = preprocessing.MinMaxScaler().fit_transform(clr_imgs_flat)
fa_clr = decomposition.FactorAnalysis(clr_projections.shape[-1],
                                      random_state=seed).fit(clr_imgs_flat)
fa_embs = fa_clr.transform(clr_imgs_flat)
fa_avg_score_dict, fa_class_score_dict = get_scores(fa_embs,
                                                    clr_labels,
                                                    mode="pca")
df2 = pd.DataFrame.from_dict(fa_avg_score_dict,
                             orient="index",
                             columns=["fa"]
                             )
df = df.join(df2)

# Now for the individual classes
fa_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(fa_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_fa.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_fa.pdf"), df_class,
            "fa")
#########################################################
# VAE
#########################################################
print("VAE", flush=True)
base_path = os.path.join(os.path.curdir, "data")
vae_file = os.path.join(base_path, "VAE_vanilla_Burgess.npz")
with np.load(vae_file, allow_pickle=True) as fnpz:
    vae_mus = np.vstack(fnpz["mus"])[rand_idx]
    vae_logvars = np.vstack(fnpz["logvars"])[rand_idx]
    vae_labels = np.vstack(fnpz["labels"])[rand_idx]
vae_projections = np.concatenate([vae_mus, vae_logvars], -1)
vae_avg_score_dict, vae_class_score_dict = get_scores(vae_projections,
                                                      vae_labels,
                                                      mode="vae")
df2 = pd.DataFrame.from_dict(vae_avg_score_dict,
                             orient="index",
                             columns=["vae"]
                             )
df = df.join(df2)

# Now for the individual classes
vae_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(vae_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_vae.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_vae.pdf"), df_class,
            "vae")

#########################################################
# UMAP
#########################################################
print("umap", flush=True)
with np.load(file_names[0], allow_pickle=True) as fnpz:
    clr_images = fnpz["images"][rand_idx]
    clr_labels = fnpz["labels"][rand_idx]
clr_imgs_flat = clr_images.reshape([clr_images.shape[0], -1])
clr_imgs_flat = preprocessing.MinMaxScaler().fit_transform(clr_imgs_flat)
umap_clr = umap.UMAP(n_components=clr_projections.shape[-1],
                     random_state=seed).fit(clr_imgs_flat)
umap_embs = umap_clr.transform(clr_imgs_flat)
umap_avg_score_dict, umap_class_score_dict = get_scores(umap_embs,
                                                        clr_labels,
                                                        mode="pca")
df2 = pd.DataFrame.from_dict(umap_avg_score_dict,
                             orient="index",
                             columns=["umap"]
                             )
df = df.join(df2)

# Now for the individual classes
umap_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(umap_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_umap.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_umap.pdf"), df_class,
            "umap")

#########################################################
# Random
#########################################################
print("rand", flush=True)
rand_embs = rng.uniform(0, 1, clr_projections.shape)
rand_avg_score_dict, rand_class_score_dict = get_scores(rand_embs,
                                                        clr_labels,
                                                        mode="rand")
df2 = pd.DataFrame.from_dict(rand_avg_score_dict,
                             orient="index",
                             columns=["rand"]
                             )
df = df.join(df2)
df.to_pickle(os.path.join(data_out_path, "global_avg_scores.pkl"))
save_to_pdf(os.path.join(data_out_path, "global_avg_scores.pdf"), df,
            "Global Averages")
# Now for the individual classes
rand_class_score_dict.update({"Index Title": relevant_classes})
df_class = pd.DataFrame.from_dict(rand_class_score_dict,
                                  orient="columns",
                                  ).set_index('Index Title')
df_class.to_pickle(os.path.join(data_out_path,
                                "micro_avg_scores_rand.pkl")
                   )
save_to_pdf(os.path.join(data_out_path,
                         "micro_avg_scores_rand.pdf"), df_class,
            "rand")

## Prepare a nice printout
# Sort for the f1 of the lin classifier because:
# From: Chen, T., Kornblith, S., Norouzi, M. & Hinton, G.
# A simple framework for contrastive learning of visual representations.
# arXiv (2020).:
# To evaluate the learned representations, we follow the widely used linear
# evaluation protocol (Zhang et al., 2016; Oord et al., 2018; Bachman et al.,
# 2019; Kolesnikov et al., 2019), where a linear classifier is trained on
# top of the frozen base network, and test accuracy is used as a proxy
# for representation quality.

# Since accuracy is artificially high in our case due to the high number
# of true negatives, we use the f1 score for sorting
# Print out f1, prec, rec, overlap, ami and f1 score sorted with
# the linacc sorting
dfs = df.loc[["lin_class_f1", "lin_class_prec", "lin_class_rec",
              "sim_overlap"]].sort_values(by="lin_class_f1", axis=1).T
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # Only CLR
    print(dfs.loc[["clr" in x for x in dfs.index]], flush=True)
    # Only other embedding stragies
    print(dfs.loc[["clr" not in x for x in dfs.index]], flush=True)
