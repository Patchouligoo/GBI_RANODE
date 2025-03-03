import numpy as np
import json
from array import array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from src.utils.utils import NumpyEncoder

def fit_likelihood(x_values, y_values_mean, y_values_std, w_true, events_num, output_path, logbased=True):

    x_values = x_values.reshape(-1, 1)

    # define the kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1e-3, (1e-5, 1e2)) #+ WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e+1))
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

    # ------------------------------- 95% CI of max likelihood -------------------------------
    # given x_pred, y_pred, and 95CI likelihood drop = np.log(2)/events_num, find the first left
    # intersection of max likelihood - drop with y_pred as the lower bound of the 95% CI
    CI_95_likelihood = max_likelihood - np.log(2)/events_num

    # arg_max_y_lower_bound = np.argmax(y_lower_bound)
    # arg_max_y_upper_bound = np.argmax(y_upper_bound)
    # max_y_lower_bound = y_lower_bound[arg_max_y_lower_bound]
    # CI_95_likelihood_lower = max_y_lower_bound - np.log(2)/events_num
    # # find intersection of lower CI with y upper bound as range of mu values
    # mu_lowerbound_index = np.argmin(np.abs(y_upper_bound[0:arg_max_y_upper_bound] - CI_95_likelihood_lower)) if arg_max_y_upper_bound > 0 else 0
    # mu_upperbound_index = np.argmin(np.abs(y_upper_bound[arg_max_y_upper_bound:] - CI_95_likelihood_lower)) if arg_max_y_upper_bound < len(y_upper_bound) else len(y_upper_bound) - 1

    # if logbased:
    #     mu_lowerbound = np.power(10, x_pred[0:arg_max_y_upper_bound][mu_lowerbound_index]) if arg_max_y_upper_bound > 0 else np.power(10, x_pred[0])
    #     mu_upperbound = np.power(10, x_pred[arg_max_y_upper_bound:][mu_upperbound_index]) if arg_max_y_upper_bound < len(x_pred) else np.power(10, x_pred[-1])
    # else:
    #     mu_lowerbound = x_pred[0:arg_max_y_upper_bound][mu_lowerbound_index] if arg_max_y_upper_bound > 0 else x_pred[0]
    #     mu_upperbound = x_pred[arg_max_y_upper_bound:][mu_upperbound_index] if arg_max_y_upper_bound < len(x_pred) else x_pred[-1]

    # ---------------------------------------------------------------------------------------

    with PdfPages(output_path) as pdf:
        f = plt.figure(figsize=(10, 8))
        plt.scatter(x_values.flatten(), y_values_mean, label='test points', color='black')
        plt.errorbar(x_values.flatten(), y_values_mean, yerr=y_values_std, fmt='o', color='black')
        plt.plot(x_pred, y_pred, label='fit func', color='red')
        plt.fill_between(x_pred, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='red')

        # plot 95% CI and the peak
        plt.scatter([x_pred[arg_max_likelihood]], [max_likelihood], color='red', label=f'peak $\mu$ {mu_pred:.4f}')  # peak w value
        plt.axhline(y=CI_95_likelihood, color='blue', linestyle='--')

        # if logbased:
        #     plt.scatter([np.log10(mu_lowerbound), np.log10(mu_upperbound)], [CI_95_likelihood_lower, CI_95_likelihood_lower], color='blue', label=f'95% CI of $\mu$ [{mu_lowerbound:.4f}, {mu_upperbound:.4f}]')
        # else:
        #     plt.scatter([mu_lowerbound, mu_upperbound], [CI_95_likelihood_lower, CI_95_likelihood_lower], color='blue', label=f'95% CI of $\mu$ [{mu_lowerbound:.4f}, {mu_upperbound:.4f}]')

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
        "true_mu": mu_true,
        "best_model_index": best_model_index,
        "x_pred": x_pred,
        "y_pred": y_pred,
        "sigma": sigma,
        "CI_95_likelihood": CI_95_likelihood,
        "CI_95_likelihood_drop": np.log(2)/events_num,
        "x_raw": x_values.flatten(),
        "y_raw": y_values_mean,
        "y_raw_std": y_values_std,
    }

    return output_metadata


def combined_fitting(input_dict, output_path):

    true_mu = np.nan
    mu_pred_list = []
    CI_95_likelihood_drop = np.nan
    x_values = None
    y_values_list = []
    sigma_values_list = []
    x_raw = []
    y_raw_list = []
    y_raw_std_list = []

    for key, value in input_dict.items():
        true_mu = value["true_mu"]
        mu_pred_list.append(value["mu_pred"])
        CI_95_likelihood_drop = value["CI_95_likelihood_drop"]
        x_values = value["x_pred"]
        y_values_list.append(value["y_pred"])
        sigma_values_list.append(value["sigma"])
        x_raw = value["x_raw"]
        y_raw_list.append(value["y_raw"])
        y_raw_std_list.append(value["y_raw_std"])

    x_values = np.array(x_values)
    y_values_list = np.array(y_values_list)
    sigma_values_list = np.array(sigma_values_list)
    x_raw = np.array(x_raw)
    y_raw_list = np.array(y_raw_list)
    y_raw_std_list = np.array(y_raw_std_list)

    # combine all predictions into final fitting


    # --------------------- method 1: average the likelihood pred ---------------------
    # weight_from_error = 1 / (sigma_values_list ** 2)
    # y_values_mean = np.sum(y_values_list * weight_from_error, axis=0) / np.sum(weight_from_error, axis=0)
    # y_values_std = np.sqrt(1 / np.sum(weight_from_error, axis=0))
    # # y_values_2nd_moment = np.sum((y_values_list ** 2) * weight_from_error, axis=0) / np.sum(weight_from_error, axis=0)
    # # y_values_std = np.sqrt(y_values_2nd_moment - y_values_mean ** 2)

    # arg_max_likelihood_combined = np.argmax(y_values_mean)
    # max_likelihood_combined = y_values_mean[arg_max_likelihood_combined]
    # CI_95_likelihood = max_likelihood_combined - CI_95_likelihood_drop
    # mu_pred_combined = np.power(10, x_values[arg_max_likelihood_combined])

    # --------------------- method 2: average each original scan points ---------------------
    # weight_from_error = 1 / (y_raw_std_list ** 2)
    # y_values_mean = np.sum(y_raw_list * weight_from_error, axis=0) / np.sum(weight_from_error, axis=0)
    # y_values_2nd_moment = np.sum((y_raw_list ** 2) * weight_from_error, axis=0) / np.sum(weight_from_error, axis=0)
    # y_values_std = np.sqrt(y_values_2nd_moment - y_values_mean ** 2)
    # y_values_std = np.sqrt(1 / np.sum(weight_from_error, axis=0))
    y_values_mean = np.mean(y_raw_list, axis=0)
    y_values_std = np.mean(y_raw_std_list, axis=0)

    print(f"y_values_mean: {y_values_mean}")
    print(f"y_values_std: {y_values_std}")

    # use y_values_mean and std in the final fitting
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1e-3, (1e-5, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_values_std**2, n_restarts_optimizer=100)
    gp.fit(x_raw.reshape(-1, 1), y_values_mean)

    x_pred = np.linspace(x_raw.min(), x_raw.max(), 1001).reshape(-1, 1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    x_pred = x_pred.flatten()
    y_pred = y_pred.flatten()
    sigma = sigma.flatten()

    y_lower_bound = y_pred - 1.96 * sigma
    y_upper_bound = y_pred + 1.96 * sigma

    arg_max_likelihood_combined = np.argmax(y_pred)
    max_likelihood_combined = y_pred[arg_max_likelihood_combined]
    CI_95_likelihood = max_likelihood_combined - CI_95_likelihood_drop
    mu_pred_combined = np.power(10, x_pred[arg_max_likelihood_combined])

    with PdfPages(output_path["coarse_scan_plot"].path) as pdf:

        f = plt.figure(figsize=(10, 8))
        for i in range(len(mu_pred_list)):
            arg_max_likelihood_i = np.argmax(y_values_list[i])
            max_likelihood_i = y_values_list[i][arg_max_likelihood_i]
            mu_pred_i = np.power(10, x_values[arg_max_likelihood_i])

            plt.plot(x_values, y_values_list[i], label=f'fit func {i}', color='red')
            plt.fill_between(x_values, y_values_list[i] - 1.96 * sigma_values_list[i], y_values_list[i] + 1.96 * sigma_values_list[i], alpha=0.2, color='red')
            plt.scatter([x_values[arg_max_likelihood_i]], [max_likelihood_i], color='red')  # peak w value

        plt.title(f'Combined Likelihood fit at true $\mu$ {true_mu:.4f}')
        plt.xlabel('$log_{10}(\mu)$')
        plt.ylabel('likelihood')
        
        for i in range(len(mu_pred_list)):
            plt.scatter(x_raw, y_raw_list[i], color='black')
            plt.errorbar(x_raw, y_raw_list[i], yerr=y_raw_std_list[i], fmt='o', color='black')

        # true mu
        plt.axvline(x=np.log10(true_mu), color='black', linestyle='--', label=f'true $\mu$ {true_mu:.4f}')

        plt.legend()
        pdf.savefig(f)
        plt.close() 
        
        # final fitting
        f = plt.figure(figsize=(10, 8))
        plt.plot(x_pred, y_pred, label='fit func', color='red')
        plt.fill_between(x_pred, y_lower_bound, y_upper_bound, alpha=0.2, color='red')
        plt.scatter(x_raw, y_values_mean, label='test points', color='black')
        plt.errorbar(x_raw, y_values_mean, yerr=y_values_std, fmt='o', color='black')

        plt.scatter([x_pred[arg_max_likelihood_combined]], [max_likelihood_combined], color='red', label=f'peak $\mu$ {mu_pred_combined:.4f}')

        plt.axvline(x=np.log10(true_mu), color='black', linestyle='--', label=f'true $\mu$ {true_mu:.4f}')

        plt.axhline(y=CI_95_likelihood, color='blue', linestyle='--', label=f'95% CI of $\mu$')

        plt.title(f'Combined Likelihood fit at true $\mu$ {true_mu:.4f}')
        plt.xlabel('$log_{10}(\mu)$')
        plt.ylabel('likelihood')

        plt.legend()
        pdf.savefig(f)
        plt.close()

    # # --------------------- method 3: simply average each original scan points ---------------------
    # # combined all predictions into final fitting
    # y_values_mean = np.mean(y_values_list, axis=0)
    # # y_values_std = np.std(y_values_list, axis=0)
    # arg_max_likelihood_combined = np.argmax(y_values_mean)
    # max_likelihood_combined = y_values_mean[arg_max_likelihood_combined]
    # CI_95_likelihood = max_likelihood_combined - CI_95_likelihood_drop
    # mu_pred_combined = np.power(10, x_values[arg_max_likelihood_combined])

    # # plot all individual fittings in one plot
    # with PdfPages(output_path["coarse_scan_plot"].path) as pdf:
    #     f = plt.figure(figsize=(10, 8))
    #     for i in range(len(mu_pred_list)):
    #         arg_max_likelihood_i = np.argmax(y_values_list[i])
    #         max_likelihood_i = y_values_list[i][arg_max_likelihood_i]
    #         mu_pred_i = np.power(10, x_values[arg_max_likelihood_i])

    #         plt.plot(x_values, y_values_list[i], label=f'fit func {i}', color='red')
    #         plt.fill_between(x_values, y_values_list[i] - 1.96 * sigma_values_list[i], y_values_list[i] + 1.96 * sigma_values_list[i], alpha=0.2, color='red')
    #         plt.scatter([x_values[arg_max_likelihood_i]], [max_likelihood_i], color='red')  # peak w value

    #     plt.title(f'Combined Likelihood fit at true $\mu$ {true_mu:.4f}')
    #     plt.xlabel('$log_{10}(\mu)$')
    #     plt.ylabel('likelihood')

    #     for i in range(len(mu_pred_list)):
    #         plt.scatter(x_raw, y_raw_list[i], color='black')
    #         plt.errorbar(x_raw, y_raw_list[i], yerr=y_raw_std_list[i], fmt='o', color='black')

    #     # true mu
    #     plt.axvline(x=np.log10(true_mu), color='black', linestyle='--', label=f'true $\mu$ {true_mu:.4f}')

    #     plt.legend()
    #     pdf.savefig(f)
    #     plt.close()

    #     # final fitting
    #     f = plt.figure(figsize=(10, 8))
    #     plt.plot(x_values, y_values_mean, label='fit func', color='red')
    #     plt.fill_between(x_values, y_values_mean - 1.96 * y_values_std, y_values_mean + 1.96 * y_values_std, alpha=0.2, color='red')
    #     plt.scatter([x_values[arg_max_likelihood_combined]], [max_likelihood_combined], color='red', label=f'peak $\mu$ {mu_pred_combined:.4f}')

    #     plt.axvline(x=np.log10(true_mu), color='black', linestyle='--', label=f'true $\mu$ {true_mu:.4f}')

    #     plt.axhline(y=CI_95_likelihood, color='blue', linestyle='--', label=f'95% CI of $\mu$')

    #     plt.title(f'Combined Likelihood fit at true $\mu$ {true_mu:.4f}')
    #     plt.xlabel('$log_{10}(\mu)$')
    #     plt.ylabel('likelihood')

    #     plt.legend()
    #     pdf.savefig(f)
    #     plt.close()


    peak_info = {
        "mu_pred": mu_pred_combined,
        "true_mu": true_mu,
        "x_pred": x_values,
        "y_pred": y_values_mean,
        "CI_95_likelihood": CI_95_likelihood,
    }

    with open(output_path["peak_info"].path, 'w') as f:
        json.dump(peak_info, f, cls=NumpyEncoder)
