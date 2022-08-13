import os
import cv2
import math
import h5py
import numpy as np
import tensorflow as tf
from scipy import ndimage
import scipy.constants as cs
from skimage.transform import resize

# Constants and lists
images = []
labels = []
photon_energy = []
bunch_id = []
radius = []
cart_images = []
_EV = cs.physical_constants["electron volt"][0]


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


def draw_black_hole_in_middle(img):
    img = np.squeeze(img)
    return np.expand_dims(blur_circle_img * img, -1)


# Dims
img_dim = 1035
target_dim = 256

# Draw the black circle in the middle
blacks = np.ones([img_dim, img_dim])
center = int(np.shape(blacks)[0] / 2)
rad_circ = int(0.30 * center)
circle_img = cv2.circle(blacks, (center, center), rad_circ, 0, -1)
blur_circle_img = ndimage.gaussian_filter(circle_img, sigma=1.5)

# Crop sizes for the images
# For the Polar Images
pol_crop_size = np.round(np.cos(np.pi / 4) * 0.99 * img_dim)
lr_cut = np.ceil((img_dim - pol_crop_size) / 2).astype(int)
# For the Cartesian Images
x_dims = (int(rad_circ * 2), int(np.sqrt(lr_cut) * 2))
ci_y_cut = np.ceil(sum(x_dims) / 2).astype(int)

# Load Raw Data
path = os.path.join(os.path.curdir, "data")
with np.load(os.path.join(path, "lbl_radius.npz")) as f:
    rad_label = f["label"]
    radii = f["radius"]

# Load data file from PRE manuscript
# You can download the file here: https://www.cxidb.org/id-94.html
# Put it somewhere and adjust the path below
h5_file_path = os.path.join(path, "helium_nanodroplets_with_labels_zimmermann.cxi")
with h5py.File(h5_file_path, "r") as f:
    for ii in range(1, 7265):
        # Identifier in cxi file
        entry = "entry_{:01.0f}".format(ii)
        # label information
        label = np.array(f["/{}/note_1/data".format(entry)][()]).flat.__array__().astype(np.int64)
        # Only Non-Round and Non-Empty images
        if label[1] == 0 and label[-1] == 0:
            # Sanity check
            assert (rad_label[ii - 1] == label).all()
            # Write out label
            labels.append(label)
            # Get image
            image = f["/{}/instrument_1/detector_1/data".format(entry)][()].astype(
                np.float32)
            # Draw black hole in it
            image = draw_black_hole_in_middle(image)
            # Project to cartesian and cut to square dimensions
            ci = tf.expand_dims(polar_to_cartesian(
                image.squeeze()), -1).numpy()[ci_y_cut:-ci_y_cut, x_dims[0]:-x_dims[-1]]
            # Cut polar image to square dimensions
            image = image[lr_cut:-lr_cut, lr_cut:-lr_cut]
            # Scale between 0 and 1 and resize to target_dim
            ci = norm_img(resize(ci, (target_dim, target_dim)))
            image = norm_img(resize(image, (target_dim, target_dim)))
            # Write out both images
            images.append(image)
            cart_images.append(ci)
            # Get Photon Energy, BunchID and Radius and write them to list
            pe = f["/{}/instrument_1/source_1/energy".format(entry)][()].astype(np.float32) / _EV
            photon_energy.append(pe)
            bi = f["/{}/experimental_identifier".format(entry)][()].astype(np.int64)
            bunch_id.append(bi)
            radius.append(radii[ii - 1])

    # Save to NPZ file
    # We drop the first image as we then have 1260 which works better with
    # batch sizes
    np.savez("static_helium_no_round_cart_all_meta.npz",
             images=images[1:], cart_images=cart_images[1:],
             labels=labels[1:], photon_energy=photon_energy[1:],
             bunch_id=bunch_id[1:], radius=radius[1:])
