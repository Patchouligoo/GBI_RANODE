import numpy as np
from array import array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

def fit_likelihood(x_values, y_values_mean, y_values_std, w_true, events_num, output_path, logbased=True):

    x_values = x_values.reshape(-1, 1)

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

    y_lower_bound = y_pred - 1.96 * sigma
    y_upper_bound = y_pred + 1.96 * sigma

    # find peak of the likelihood
    arg_max_likelihood = np.argmax(y_pred)
    max_likelihood = y_pred[arg_max_likelihood]

    if logbased:
        mu_pred = np.power(10, x_pred[arg_max_likelihood])
        mu_true = np.power(10, w_true)
    else:
        mu_pred = x_pred[arg_max_likelihood]
        mu_true = w_true
    # get the cloest mu value to the pred mu in x_values
    best_model_index = np.argmin(np.abs(x_values - x_pred[arg_max_likelihood]))

    # 95% CI of the peak
    arg_max_y_lower_bound = np.argmax(y_lower_bound)
    arg_max_y_upper_bound = np.argmax(y_upper_bound)
    max_y_lower_bound = y_lower_bound[arg_max_y_lower_bound]
    CI_95_likelihood_lower = max_y_lower_bound - np.log(2)/events_num
    # find intersection of lower CI with y upper bound as range of mu values
    mu_lowerbound_index = np.argmin(np.abs(y_upper_bound[0:arg_max_y_upper_bound] - CI_95_likelihood_lower))
    mu_upperbound_index = np.argmin(np.abs(y_upper_bound[arg_max_y_upper_bound:] - CI_95_likelihood_lower))

    if logbased:
        mu_lowerbound = np.power(10, x_pred[0:arg_max_y_upper_bound][mu_lowerbound_index])
        mu_upperbound = np.power(10, x_pred[arg_max_y_upper_bound:][mu_upperbound_index])
    else:
        mu_lowerbound = x_pred[0:arg_max_y_upper_bound][mu_lowerbound_index]
        mu_upperbound = x_pred[arg_max_y_upper_bound:][mu_upperbound_index]

    with PdfPages(output_path) as pdf:
        f = plt.figure(figsize=(10, 8))
        plt.scatter(x_values.flatten(), y_values_mean, label='test points', color='black')
        plt.errorbar(x_values.flatten(), y_values_mean, yerr=y_values_std, fmt='o', color='black')
        plt.plot(x_pred, y_pred, label='fit func', color='red')
        plt.fill_between(x_pred, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='red')

        # plot 95% CI and the peak
        plt.scatter([x_pred[arg_max_likelihood]], [max_likelihood], color='red', label=f'peak $\mu$ {mu_pred:.4f}')  # peak w value
        plt.axhline(y=CI_95_likelihood_lower, color='blue', linestyle='--')

        if logbased:
            plt.scatter([np.log10(mu_lowerbound), np.log10(mu_upperbound)], [CI_95_likelihood_lower, CI_95_likelihood_lower], color='blue', label=f'95% CI of $\mu$ [{mu_lowerbound:.4f}, {mu_upperbound:.4f}]')
        else:
            plt.scatter([mu_lowerbound, mu_upperbound], [CI_95_likelihood_lower, CI_95_likelihood_lower], color='blue', label=f'95% CI of $\mu$ [{mu_lowerbound:.4f}, {mu_upperbound:.4f}]')

        # true w value
        plt.axvline(x=w_true, color='black', linestyle='--', label=f'true $\mu$ {mu_true:.4f}')

        plt.title(f'Likelihood fit at true $\mu$ {mu_true:.4f}')

        if logbased:
            plt.xlabel('$log_{10}(\mu)$')
        else:
            plt.xlabel('$\mu$')

        plt.ylabel('likelihood')

        plt.legend()
        pdf.savefig(f)
        plt.close()

    output_metadata = {
        "mu_pred": mu_pred,
        "best_model_index": best_model_index,
        "mu_lowerbound": mu_lowerbound,
        "mu_upperbound": mu_upperbound,
    }

    return output_metadata