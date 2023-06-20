# %% [markdown]
# <center>
# 
# # [`ProxNest`](https://github.com/astro-informatics/proxnest) - __Radio Interferometry Example__ Interactive Tutorial
# ---
# Suppose we collect complete Fourier observations $y \in \mathbb{R}^M$ of some image $x \in \mathbb{R}^N$ under a forward model $\Phi = \mathsf{G}\mathsf{F}$, where $M \ll N$ and $\mathsf{G}$ / $\mathsf{F}$ are Fourier sub-sampling and transform respectively. Suppose further that our observational instrument introduces some aleoteric uncertainty which can be adequately modelled by a univariate Gaussian $n = \mathcal{N}(0, \sigma) \in \mathbb{R}^N$. In this case our measurement equation is given by $$y = \Phi x + n.$$
# 
# Under these conditions the inverse problem of infering $x$ given $y$ is heavily degenerate and thus breaks [Hadamards](https://en.wikipedia.org/wiki/Well-posed_problem) second condition: the solution is not unique and thus the inverse problem is ill-posed. Given that inferences of $x$ are degenerate it naturally makes more sense to consider the probability distribution of possible solutions; the posterior distribution. Here we use **proximal nested sampling** ([Cai *et al* 2022](https://arxiv.org/abs/2106.03646)), which allows us to sample from this posterior distribution, recovering both an estimate of $x$ and the plausibility of this estimate. Moreover, as this is a nested sampling algorithm we automatically recover the Bayesian evidence, which naturally allows us to carry out model comparison, through which one can *e.g.* determine which forward models $\Phi$ are favoured by the data, or calibrate hyper-parameters of the problem such as $\sigma$ and regularisation parameters $\lambda$.

# %%
import sys
import os
# Set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import ProxNest as pxn
import ProxNest.utils as utils
import ProxNest.sampling as sampling
import ProxNest.optimisations as optimisations
import ProxNest.operators as operators

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from matplotlib import pyplot as plt
plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"




# %% [markdown]
# ### Load an image and simulate some observations

# %%
# Load Image
dim = 64
ground_truth = np.load('../../data/galaxy_image_{}.npy'.format(dim))

# Normalise magnitude
ground_truth -= np.nanmin(ground_truth)
ground_truth /= np.nanmax(ground_truth)
ground_truth[ground_truth<0] = 0

# %% [markdown]
# ### Construct linear operators and mock simulated observations for our stated problem

# %%
np.random.seed(0)

# A mock radio imaging forward model with half of the Fourier coefficients masked
phi = operators.sensing_operators.MaskedFourier(dim, 0.3)

# A wavelet dictionary in which we can promote sparsity
psi = operators.wavelet_operators.db_wavelets(["db6"], 2, (dim, dim))

# %%
# Simulate mock noisy observations y
y = phi.dir_op(ground_truth)
ISNR = 15
sigma = np.sqrt(np.mean(np.abs(y)**2)) * 10**(-ISNR/20)
n = np.random.normal(0, sigma, y.shape)

# Simulate mock noisy observations
y += n

# %% [markdown]
# ### Define all necessary parameters and posteior lambda functions

# %%
alpha = 1.
varepsilon = sigma**2
L = 1.
L_y = 0.

lamb = 0.1 * 0.5 * (1/(2 * L_y + alpha * L / varepsilon ))
delta = 0.1 * (1/3) * (1/ (L_y + 1/lamb + alpha * L / varepsilon))


# %%
print('varepsilon: ', varepsilon)
print('lamb: ', lamb)
print('delta: ', delta)


# Translate variables
delta_step_DnCNN = delta
lamb_DnCNN = lamb
gamma_DnCNN = varepsilon



# %%
delta_step_WAV = 1e-8
lamb_WAV = 5. * delta_step_WAV
gamma_WAV = 5. * delta_step_WAV


# %%
# Parameter dictionary associated with optimisation problem of resampling from the prior subject to the likelihood iso-ball
params_WAV = utils.create_parameters_dict(
           y = np.copy(y),           # Measurements i.e. data
         Phi = phi,                  # Forward model
     epsilon = 1e-3,                 # Radius of L2-ball of likelihood 
       tight = False,                # Is Phi a tight frame or not?
          nu = 1,                    # Bound on the squared-norm of Phi
         tol = 1e-12,                # Convergence tolerance of algorithm
    max_iter = 200,                  # Maximum number of iterations
     verbose = 0,                    # Verbosity level
           u = 0,                    # Initial vector for the dual problem
         pos = True,                 # Positivity flag
     reality = True                  # Reality flag
)

params_DnCNN = utils.create_parameters_dict(
           y = np.copy(y),           # Measurements i.e. data
         Phi = phi,                  # Forward model
     epsilon = 1e-3,                 # Radius of L2-ball of likelihood 
       tight = False,                # Is Phi a tight frame or not?
          nu = 1,                    # Bound on the squared-norm of Phi
         tol = 1e-12,                # Convergence tolerance of algorithm
    max_iter = 200,                  # Maximum number of iterations
     verbose = 0,                    # Verbosity level
           u = 0,                    # Initial vector for the dual problem
         pos = True,                 # Positivity flag
     reality = True                  # Reality flag
)

# Options dictionary associated with the overall sampling algorithm
options_DnCNN = utils.create_options_dict(
    samplesL = 2e3,                  # Number of live samples
    samplesD = 4e4,                  # Number of discarded samples 
    thinning = 1e2,                  # Thinning factor (to mitigate correlations)
       delta = delta_step_DnCNN, #1e-8,    # Discretisation stepsize
       lamb = lamb_DnCNN,
        burn = 1e3,                  # Number of burn in samples
       sigma = sigma,                # Noise standard deviation of degraded image
       gamma = gamma_DnCNN,             # Gamma parameter of the prior term. Using noise variance
)

options_WAV = utils.create_options_dict(
    samplesL = 2e3,                  # Number of live samples
    samplesD = 4e4,                  # Number of discarded samples 
    thinning = 1e2,                  # Thinning factor (to mitigate correlations)
       delta = delta_step_WAV, #1e-8,    # Discretisation stepsize
       lamb = lamb_WAV,
        burn = 1e3,                  # Number of burn in samples
       sigma = sigma,                # Noise standard deviation of degraded image
       gamma = gamma_WAV,             # Gamma parameter of the prior term. Using noise variance
)


# %%
# Regularisation parameter
delta = 1e5

# Lambda functions to evaluate cost function
LogLikeliL = lambda sol : - np.linalg.norm(y-phi.dir_op(sol))**2/(2*sigma**2)

# Lambda function for L1-norm wavelet prior backprojection steps
proxH_WAV = lambda x, T : operators.proximal_operators.l1_projection(x, T, delta, Psi=psi)


# Lambda function for L2-ball likelihood projection during resampling
proxB_WAV = lambda x, tau: optimisations.l2_ball_proj.sopt_fast_proj_B2(x, tau, params_WAV)

# Lambda function for L2-ball likelihood projection during resampling
proxB_DnCNN = lambda x, tau: optimisations.l2_ball_proj.sopt_fast_proj_B2(x, tau, params_DnCNN)


# %%
# Saved dir of the model in SavedModel format
saved_model_path = '/disk/xray0/tl3/repos/lexci_models/DnCNN/snr_15_model.pb'
# Load DnCNN denoiser prox
proxH_DnCNN = pxn.operators.learned_operators.prox_DnCNN(saved_model_path)


# %% [markdown]
# ### Select a starting position $X_0$ and execute the sampling method

# %%
# Create a 'dirty image' starting position
X0 = np.abs(phi.adj_op(np.copy(y)))

# %%
save_fig_dir = '/disk/xray0/tl3/repos/proxnest/debug/figs/'
save_var_dir = '/disk/xray0/tl3/repos/proxnest/debug/saved_vars/'

save_prefix_WAV = 'radio_WAV_delta_{:.1e}_lamb_{:.1e}_gamma_{:.1e}'.format(
    delta_step_WAV, lamb_WAV, gamma_WAV
)

save_prefix_DnCNN = 'radio_DnCNN_delta_{:.1e}_lamb_{:.1e}_gamma_{:.1e}'.format(
    delta_step_DnCNN, lamb_DnCNN, gamma_DnCNN
)


# %% [markdown]
# # DnCNN model

# %%

# Perform proximal nested sampling
NS_BayEvi_DnCNN, NS_Trace_DnCNN = sampling.proximal_nested.ProxNestedSampling(
    np.copy(X0), LogLikeliL, proxH_DnCNN, proxB_DnCNN, params_DnCNN, options_DnCNN
)


# %%
print(NS_BayEvi_DnCNN)

# %%
# DnCNN

images = [ground_truth, X0, NS_Trace_DnCNN['DiscardPostMean']]
labels = ["Truth", "Dirty", "Posterior mean"]

fig, axs = plt.subplots(1,3, figsize=(20,8), dpi=400)
for i in range(3):
    axs[i].imshow(images[i], cmap='afmhot', vmax=np.nanmax(images), vmin=np.nanmin(images))
    if i > 0:   
        stats_str = ' (PSNR: {}, SSIM: {})'.format(
            round(psnr(ground_truth, images[i], data_range=ground_truth.max()-ground_truth.min()), 2),
            round(ssim(ground_truth, images[i], data_range=ground_truth.max()-ground_truth.min()), 2)
            )
        labels[i] += stats_str
    axs[i].set_title(labels[i], fontsize=16)
    axs[i].axis('off')

plt.savefig('{:s}{:s}_reconstruction_plot.pdf'.format(save_fig_dir, save_prefix_DnCNN))
plt.show()

dirty_DnCNN_SNR = psnr(ground_truth, X0, data_range=ground_truth.max()-ground_truth.min())
post_mean_DnCNN_SNR = psnr(ground_truth, NS_Trace_DnCNN['DiscardPostMean'], data_range=ground_truth.max()-ground_truth.min())


# %%
save_dict_DnCNN = {
    'ground_truth': ground_truth,
    'X0': X0,
    'dirty_SNR': dirty_DnCNN_SNR,
    'post_mean_SNR': post_mean_DnCNN_SNR,
    'NS_BayEvi': NS_BayEvi_DnCNN,
    'NS_Trace': NS_Trace_DnCNN,
    'options': options_DnCNN,
}

try:
    np.save('{:s}{:s}_saved_vars.pdf'.format(save_var_dir, save_prefix_DnCNN), save_dict_DnCNN, allow_pickle=True)
except Exception as e:
    print('cannot save variables')
    print(e)


# %% [markdown]
# # WAV model

# %%

# Perform proximal nested sampling
NS_BayEvi_WAV, NS_Trace_WAV = sampling.proximal_nested.ProxNestedSampling(
    np.copy(X0), LogLikeliL, proxH_WAV, proxB_WAV, params_WAV, options_WAV
)



# %%
print(NS_BayEvi_WAV)


# %%
# WAV

images = [ground_truth, X0, NS_Trace_WAV['DiscardPostMean']]
labels = ["Truth", "Dirty", "Posterior mean"]

fig, axs = plt.subplots(1,3, figsize=(20,8), dpi=400)
for i in range(3):
    axs[i].imshow(images[i], cmap='afmhot', vmax=np.nanmax(images), vmin=np.nanmin(images))
    if i > 0:   
        stats_str = ' (PSNR: {}, SSIM: {})'.format(
            round(psnr(ground_truth, images[i], data_range=ground_truth.max()-ground_truth.min()), 2),
            round(ssim(ground_truth, images[i], data_range=ground_truth.max()-ground_truth.min()), 2)
            )
        labels[i] += stats_str
    axs[i].set_title(labels[i], fontsize=16)
    axs[i].axis('off')


plt.savefig('{:s}{:s}_reconstruction_plot.pdf'.format(save_fig_dir, save_prefix_WAV))
plt.show()
# plt.close()

dirty_WAV_SNR = psnr(ground_truth, X0, data_range=ground_truth.max()-ground_truth.min())
post_mean_WAV_SNR = psnr(ground_truth, NS_Trace_WAV['DiscardPostMean'], data_range=ground_truth.max()-ground_truth.min())



# %%

save_dict_WAV = {
    'ground_truth': ground_truth,
    'X0': X0,
    'dirty_SNR': dirty_WAV_SNR,
    'post_mean_SNR': post_mean_WAV_SNR,
    'NS_BayEvi': NS_BayEvi_WAV,
    'NS_Trace': NS_Trace_WAV,
    'options': options_WAV,
}

try:
    np.save('{:s}{:s}_saved_vars.pdf'.format(save_var_dir, save_prefix_WAV), save_dict_WAV, allow_pickle=True)
except Exception as e:
    print('cannot save variables')
    print(e)



# %%


old_stdout = sys.stdout
log_file = open('{:s}{:s}_log.log'.format(save_var_dir, save_prefix_WAV), 'w')
sys.stdout = log_file
print('Starting the log file.\n')


print('NS_BayEvi_DnCNN: ', NS_BayEvi_DnCNN)
print('NS_BayEvi_WAV: ', NS_BayEvi_WAV)

print('dirty_DnCNN_SNR: ', dirty_DnCNN_SNR)
print('post_mean_DnCNN_SNR: ', post_mean_DnCNN_SNR)

print('dirty_WAV_SNR: ', dirty_WAV_SNR)
print('post_mean_WAV_SNR: ', post_mean_WAV_SNR)


## Close log file
print('\n Good bye..')
sys.stdout = old_stdout
log_file.close()



# %%


# %%



