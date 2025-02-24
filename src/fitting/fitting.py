import numpy as np
from array import array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

def fit_likelihood(x_values, y_values, w_true, events_num, output_path):

    x_values = x_values.reshape(-1, 1)
    y_values_mean = np.mean(y_values, axis=-1)
    y_values_std = np.std(y_values, axis=-1)

    # define the kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) #+ WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_values_std**2, n_restarts_optimizer=100)
    gp.fit(x_values, y_values_mean)

    # --- Make Predictions on a Fine Grid ---
    x_pred = np.linspace(x_values.min(), x_values.max(), 1001).reshape(-1, 1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    # --- Plot the Fit ---
    x_pred = x_pred.flatten()
    y_pred = y_pred.flatten()
    sigma = sigma.flatten()

    # find peak of the likelihood
    arg_max_likelihood = np.argmax(y_pred)
    max_likelihood = y_pred[arg_max_likelihood]
    mu_pred = np.power(10, x_pred[arg_max_likelihood])
    mu_true = np.power(10, w_true)
    CI_95_likelihood = max_likelihood - np.log(2)/events_num

    with PdfPages(output_path) as pdf:
        f = plt.figure(figsize=(10, 8))
        plt.scatter(x_values.flatten(), y_values_mean, label='test points', color='black')
        plt.errorbar(x_values.flatten(), y_values_mean, yerr=y_values_std, fmt='o', color='black')
        plt.plot(x_pred, y_pred, label='fit func', color='red')
        plt.fill_between(x_pred, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='red')

        # plot 95% CI of the peak
        
        plt.axhline(y=CI_95_likelihood, color='red', linestyle='--', label='95% CI of $\mu$')

        plt.scatter([x_pred[arg_max_likelihood]], [max_likelihood], color='red', label=f'peak $\mu$ {mu_pred:.4f}')  # peak w value
        
        # true w value
        plt.axvline(x=w_true, color='black', linestyle='--', label=f'true $\mu$ {mu_true:.4f}')

        plt.title(f'Likelihood fit at true $\mu$ {mu_true:.4f}')

        plt.xlabel('$log_{10}(\mu)$')
        plt.ylabel('likelihood')

        plt.legend()
        pdf.savefig(f)
        plt.close()
