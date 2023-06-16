import tensorflow as tf
from functools import partial
import os


def load_DnCNN(saved_model_path):
    if not os.path.isdir(saved_model_path):
        raise FileNotFoundError('Directory with model does not exist')
    return tf.saved_model.load(saved_model_path)

def _DnCNN_denoise(noisy_img, sigma, tf_model):
    """Apply DnCNN denoising

    Args:
        noisy_img (np.ndarray): Image to denoise. Should be [Im_x, Im_y] dimensions.
    
    Returns:
        np.ndarray: Denoised version of `noisy_img`.
    """
    # Convert to tensor with correct dtype
    tf_noisy_img = tf.convert_to_tensor(noisy_img, dtype=tf.float32)
    # Add requried dimensions [Batch, Im_x, Im_y, C]
    tf_noisy_img = tf_noisy_img[tf.newaxis,...,tf.newaxis]
    # Denoise
    tf_denoised_img = tf_model(tf_noisy_img)
    # Remove extra dimensions and convert to numpy
    return tf.squeeze(tf_denoised_img, axis=[0, 3]).numpy()

def prox_DnCNN(saved_model_path):
    """Return callable DnCNN denoiser
    """
    return partial(_DnCNN_denoise, tf_model=load_DnCNN(saved_model_path))

