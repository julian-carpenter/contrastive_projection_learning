import os
import math
import h5py
import glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.constants as cs
from skimage.transform import resize


def get_intersect_file_list(run_list_a, run_list_b, gas_list):
    """
    Return only these Runs that appear in both lists.
    This is to avoid that one list contains a Run nr. that is not
    in the other list.
    """
    runs_a = [int(os.path.basename(x)[3:7]) for x in run_list_a]
    runs_b = [int(os.path.basename(x)[3:7]) for x in run_list_b]
    runs_c = list(gas_meta[:, 0].astype(int))
    intersection = np.intersect1d(np.intersect1d(runs_a, runs_b), runs_c)
    idx_a = [runs_a.index(int(x)) for x in intersection]
    idx_b = [runs_b.index(int(x)) for x in intersection]
    idx_c = [runs_c.index(int(x)) for x in intersection]

    return idx_a, idx_b, idx_c


# Function defs
def norm_img(data, min_new=0, max_new=1):
    """Normalize data to max_new and  min_new."""
    if min_new != max_new:
        data = np.array(data).astype(np.float32)
        min_new = np.array(min_new).astype(np.float32)
        max_new = np.array(max_new).astype(np.float32)
        max_old = np.max(data)
        min_old = np.min(data)

        quot = np.divide(np.subtract(max_new, min_new),
                         np.subtract(max_old, min_old))

        mult = np.multiply(quot, np.subtract(data, max_old))
        data = np.add(mult, max_new)
        return data
    else:
        raise ArithmeticError


def polar_to_cartesian(img):
    """
    Polar to cartesian transformation in tensorflow -> NN Interpolation
    """
    r = img.shape[0]
    i = tf.tile(tf.cast(tf.linspace(0.0, r - 1, r)[:, None], tf.float32), [1, r])
    j = tf.tile(tf.cast(tf.linspace(0.0, r - 1, r)[None, :], tf.float32), [r, 1])

    i_phi = i / (r - 1) * 2 * math.pi
    j_rho = j / (r - 1)
    rp_x = j_rho * tf.cos(i_phi)
    rp_y = j_rho * tf.sin(i_phi)
    y_i = .5 * (r - 1) * (1 - rp_y)
    x_j = .5 * (r - 1) * (1 + rp_x)
    i_cartesian = tf.cast(tf.round(y_i), tf.int32)
    j_cartesian = tf.cast(tf.round(x_j), tf.int32)
    new_map = tf.stack([i_cartesian, j_cartesian], axis=-1)
    img = tf.gather_nd(img, new_map)
    return img


# Constants and lists
images = []
cart_images = []
meta = {}
_EV = cs.physical_constants["electron volt"][0]

# Dims
img_dim = 512
target_dim = 256
center = int(img_dim / 2)
rad_circ = int(0.1 * center)

# Crop sizes for the images
# For the Polar Images
pol_crop_size = np.round(np.cos(np.pi / 4) * img_dim * 0.75)
lr_cut = np.ceil((img_dim - pol_crop_size) / 2).astype(int)
# For the Cartesian Images
x_dims = (int(rad_circ * 2), int(np.sqrt(lr_cut) * 2))
ci_y_cut = np.ceil(sum(x_dims) / 2).astype(int)

# Load Raw Data
path_images = os.path.join(os.sep, "home", "nux", "Data", "2022", "Maloja", "rescaled_data", "all_metadata")
path_pe_meta = os.path.join(os.sep, "home", "nux", "Data", "2022", "Maloja", "rescaled_data", "all_additional")
path_gas_meta = os.path.join(os.sep, "scratch", "jzimmermann", "maloja-analysis", "data")
out_path = os.path.join(os.sep, "scratch", "jzimmermann", "datasets")
# out_path = os.path.join(os.sep, "home", "julian", "Downloads")
# path_images = os.path.join(os.sep, "home", "julian", "Downloads")
# path_pe_meta = os.path.join(os.sep, "home", "julian", "Downloads", "add")
# path_gas_meta = os.path.join(os.sep, "home", "julian", "Dropbox", "Science", "ETH", "Projects", "Maloja", "data")

csv_gas_meta_file = os.path.join(path_gas_meta, "2022_Maloja_RunsOverview.csv")
h5_image_files = glob.glob(os.path.join(path_images, "Run*.h5"))
h5_pe_files = glob.glob(os.path.join(path_pe_meta, "Run*_additional.h5"))
# The gas csv is a four-column file using this header:
# Run, Xe Gas (%), Source Pressure (bar), Source Temp (K)
gas_meta = np.genfromtxt(csv_gas_meta_file, delimiter=",", skip_header=1)

# Get the runs for which image data and meta data is available
h5_image_files_idx, h5_pe_files_idx, gas_meta_idx = get_intersect_file_list(h5_image_files,
                                                                            h5_pe_files,
                                                                            gas_meta)
h5_image_files = np.array(h5_image_files)[h5_image_files_idx]
h5_pe_files = np.array(h5_pe_files)[h5_pe_files_idx]
gas_meta = gas_meta[gas_meta_idx]

for img_file, meta_file, gas_info in tqdm(zip(h5_image_files, h5_pe_files, gas_meta)):
    with h5py.File(img_file, "r") as f_img, h5py.File(meta_file, "r") as f_meta:
        file_keys = list(f_img)
        if "metadata" in file_keys:
            meta_fields = list(f_img["metadata"])
        else:
            # raise KeyError("'acq' not in run {}".format(os.path.basename(img_file)))
            print("'metadata' not in run {}".format(os.path.basename(img_file)))
            continue

        if "acq" in file_keys:
            nr_images = len(list(f_img["acq"]))
        else:
            # raise KeyError("'acq' not in run {}".format(os.path.basename(img_file)))
            print("'acq' not in run {}".format(os.path.basename(img_file)))
            continue

        # Synchronize additional PSI metadata
        pids_im = list(f_img["metadata/pid"].astype(int))
        pids_pe = list(f_meta["id"])
        sorting_idx = []
        for v in pids_im:
            try:
                sorting_idx.append(pids_pe.index(v))
            except ValueError:  # Ignore the missing pids
                continue

        # We take only those images whose integral is above a threshold
        sorted_intensities = np.sort(f_img["metadata/intensity"])[::-1]
        cutoff = np.argmin(np.abs(sorted_intensities - 3e5))
        sorted_idx = np.argsort(list(f_img["metadata/intensity"].astype(int)))[::-1][:cutoff]

        if all([x in meta for x in meta_fields]):
            # Add to existing
            # OLAF metadata
            meta.update({k: np.concatenate([meta[k], np.array(v)[sorted_idx]]) for k, v in f_img["metadata"].items()})
            # Additional meta
            pe_array = np.array(list(f_meta["photon_energy"]))
            meta.update({"photon_energy": np.concatenate(
                [meta["photon_energy"], pe_array[sorting_idx][sorted_idx]]
            )})
            # Gas meta
            meta.update({"xenon_concentration": np.concatenate(
                [meta["xenon_concentration"], len(sorted_idx) * [gas_info[1]]]
            )})
            meta.update({"source_pressure": np.concatenate(
                [meta["source_pressure"], len(sorted_idx) * [gas_info[2]]]
            )})
            meta.update({"source_temperature": np.concatenate(
                [meta["source_temperature"], len(sorted_idx) * [gas_info[3]]]
            )})

        elif 0 < sum([x in meta for x in meta_fields]) < len(meta_fields):
            # break if incomplete
            print("'metadata' incomplete in run {}".format(os.path.basename(img_file)))
            continue

        else:
            # create if first
            # OLAF metadata
            meta.update({k: np.array(v)[sorted_idx] for k, v in f_img["metadata"].items()})
            # Additional meta
            pe_array = np.array(list(f_meta["photon_energy"]))
            meta.update({"photon_energy": pe_array[sorting_idx][sorted_idx]})
            # Gas meta
            meta.update({"xenon_concentration": len(sorted_idx) * [gas_info[1]]})
            meta.update({"source_pressure": len(sorted_idx) * [gas_info[2]]})
            meta.update({"source_temperature": len(sorted_idx) * [gas_info[3]]})

        for image in np.array(f_img["data"])[sorted_idx].astype(np.float32):
            # Project to cartesian and cut to square dimensions
            ci = tf.expand_dims(polar_to_cartesian(
                image.squeeze()), -1).numpy()[ci_y_cut:-ci_y_cut, x_dims[0]:-x_dims[-1]]
            # Cut polar image to square dimensions
            image = tf.expand_dims(image[lr_cut:-lr_cut, lr_cut:-lr_cut], -1)
            # Scale between 0 and 1 and resize to target_dim
            ci = norm_img(resize(ci, (target_dim, target_dim)))
            image = norm_img(resize(image, (target_dim, target_dim)))
            # Write out both images
            images.append(image)
            cart_images.append(ci)

# f, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(np.power(images[0], 0.25))
# ax[1].imshow(np.power(cart_images[0], 0.25))
# for a in ax:
#     a.axis(False)
# f.tight_layout()
# plt.show()

# Save to NPZ file
np.savez(os.path.join(out_path, "maloja_cplr_all_meta_3e5_threshold.npz"),
         images=images, cart_images=cart_images,
         **meta)
