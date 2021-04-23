# Please place imports here.
# BEGIN IMPORTS
import numpy as np
import scipy
from scipy import ndimage
import cv2
import imageio
import nose
from nose.tools import set_trace
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x channels image with dimensions
                  matching the input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    images = np.array(images)
    N, h, w, c = images.shape
    I_flat = np.reshape(images, (N, h*w*c))
    G = np.linalg.inv(lights@lights.T)@lights@I_flat
    G = np.reshape(G, (3, h, w, c))
    ## update this to account for zeros above, this is causing a bug when calculating normals
    albedo = np.linalg.norm(G, axis=0)
    normals = np.sum(G, axis=3)
    normals = normals / np.linalg.norm(normals, axis=0)
    normals = np.transpose(normals, axes=(1, 2, 0))
    return albedo, normals


def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and scipy.ndimage.gaussian_filter are
    prohibited.  You must implement the separable kernel.  However, you may
    use functions such as cv2.filter2D or scipy.ndimage.correlate to do the actual
    correlation / convolution. Note that for images with one channel, cv2.filter2D
    will discard the channel dimension so add it back in.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) x channels image of type
                float32.
    """
    x_filter = np.expand_dims(1/16 * np.array([1, 4, 6, 4, 1]), axis=(0, 2))
    y_filter = np.expand_dims(1/16 * np.array([1, 4, 6, 4, 1]), axis=(1, 2))
    blurred_img = ndimage.convolve(image, x_filter, mode="mirror")
    blurred_img = ndimage.convolve(blurred_img, y_filter, mode="mirror")
    downsampled_image = blurred_img[::2, ::2]
    return downsampled_image


def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        up -- 2*height x 2*width x channels image of type float32.
    """
    x_filter = np.expand_dims(1/8 * np.array([1, 4, 6, 4, 1]), axis=(0, 2))
    y_filter = np.expand_dims(1/8 * np.array([1, 4, 6, 4, 1]), axis=(1, 2))
    height = image.shape[0] * 2
    width = image.shape[1] * 2
    channels = image.shape[2]
    upsized_img = np.zeros((height, width, channels))
    height_indices = np.arange(0, image.shape[0], dtype=int) * 2
    width_indices = np.arange(0, image.shape[1], dtype=int) * 2
    for ind in height_indices:
        upsized_img[ind][width_indices] = image[ind//2]
    upscaled_img = ndimage.convolve(upsized_img, x_filter, mode="mirror")
    upscaled_img = ndimage.convolve(upscaled_img, y_filter, mode="mirror")
    return upscaled_img

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    ones = np.ones((points.shape[0], points.shape[1], 1))
    world_pts = np.concatenate((points, ones), axis=2)
    P = K@Rt
    img_pts = (world_pts @ P.T)[:, :, :2]
    return img_pts


def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R' -R't |
            | 0 1 |      | 0   1   |

          p = R' * (z * x', z * y', z, 1)' - R't

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """
    corner_pts = np.array([[[0, 0, 1], [width, 0, 1]], [[0, height, 1], [width, height, 1]]])
    corner_pts = np.transpose(corner_pts, (0, 2, 1))
    K_inv = np.linalg.inv(K)
    camera_pts = depth * np.transpose((K_inv@corner_pts), (0, 2, 1))
    camera_pts = np.append(camera_pts, np.array([[[1], [1]], [[1], [1]]]), axis=-1)
    M = np.vstack((Rt, np.array([0, 0, 0, 1])))
    M_inv = np.linalg.inv(M)
    # R_prime = M_inv[:3, :3]
    # Rt_prime = M_inv[:3, 3]
    # p = (camera_pts @ R_prime) - Rt_prime
    # print(p)
    world_pts = (camera_pts@M_inv)
    world_pts = world_pts[:, :, :3] / np.expand_dims(world_pts[:, :, 3], axis=2)
    return world_pts


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    normalized = np.zeros((image.shape[0], image.shape[1], image.shape[2]*ncc_size**2))
    pad = ncc_size // 2
    padded_image = np.pad(image,
                          pad_width=((pad, pad), (pad, pad), (0, 0)), mode="constant",
                          constant_values=((0, 0), (0, 0), (0, 0)))
    count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded_image[i:i+ncc_size, j:j+ncc_size]
            #patch -= np.expand_dims(np.mean(patch, axis=(0, 1)), axis=(0, 1))
            flat_patch = np.zeros(image.shape[2]*ncc_size**2)
            flat_patch[:ncc_size**2] = np.ndarray.flatten(patch[:, :, 0]) - np.mean(np.ndarray.flatten(patch[:, :, 0]))
            flat_patch[ncc_size**2:2*ncc_size**2] = np.ndarray.flatten(patch[:, :, 1]) - np.mean(np.ndarray.flatten(patch[:, :, 1]))
            flat_patch[2*ncc_size**2:3*ncc_size**2] = np.ndarray.flatten(patch[:, :, 2]) - np.mean(np.ndarray.flatten(patch[:, :, 2]))
            n = np.linalg.norm(flat_patch)
            if n < 1e-6:
                normalized[i, j] = np.zeros(image.shape[2]*ncc_size**2)
            else:
                normalized[i, j] = flat_patch / n
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.zeros((image1.shape[0], image1.shape[1]))
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            ncc[i, j] = np.dot(image1[i,j], image2[i,j])
    return ncc