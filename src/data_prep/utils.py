import numpy as np

def get_dijetmass_ptetaphi(jets):
    jet_e = np.sqrt(jets[:,0,3]**2 + jets[:,0,0]**2*np.cosh(jets[:,0,1])**2)
    jet_e += np.sqrt(jets[:,1,3]**2 + jets[:,1,0]**2*np.cosh(jets[:,1,1])**2)
    jet_px = jets[:,0,0]*np.cos(jets[:,0,2]) + jets[:,1,0]*np.cos(jets[:,1,2])
    jet_py = jets[:,0,0]*np.sin(jets[:,0,2]) + jets[:,1,0]*np.sin(jets[:,1,2])
    jet_pz = jets[:,0,0]*np.sinh(jets[:,0,1]) + jets[:,1,0]*np.sinh(jets[:,1,1])
    mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
    return mjj

def get_dijetmass_pxyz(jets):
    jet_e = np.sqrt(jets[:,0,3]**2 + jets[:,0,0]**2 + jets[:,0,1]**2 + jets[:,0,2]**2)
    jet_e += np.sqrt(jets[:,1,3]**2 + jets[:,1,0]**2 + jets[:,1,1]**2 + jets[:,1,2]**2)
    jet_px = jets[:,0,0] + jets[:,1,0]
    jet_py = jets[:,0,1] + jets[:,1,1]
    jet_pz = jets[:,0,2] + jets[:,1,2]
    mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
    return mjj

def standardize(x, mean, std):
    return (x - mean) / std

def logit_transform(x, min_vals, max_vals):
    with np.errstate(divide='ignore', invalid='ignore'):
        x_norm = (x - min_vals) / (max_vals - min_vals)
        logit = np.log(x_norm / (1 - x_norm))
    domain_mask = ~(np.isnan(logit).any(axis=1) | np.isinf(logit).any(axis=1))
    return logit, domain_mask

def preprocess_params_fit(data):
    preprocessing_params = {}
    preprocessing_params["min"] = np.min(data[:, 1:-1], axis=0)
    preprocessing_params["max"] = np.max(data[:, 1:-1], axis=0)

    preprocessed_data_x, mask = logit_transform(data[:, 1:-1], preprocessing_params["min"], preprocessing_params["max"])
    preprocessed_data = np.hstack([data[:, 0:1], preprocessed_data_x, data[:, -1:]])[mask]

    preprocessing_params["mean"] = np.mean(preprocessed_data[:, 1:-1], axis=0)
    preprocessing_params["std"] = np.std(preprocessed_data[:, 1:-1], axis=0)

    return preprocessing_params

def preprocess_params_transform(data, params):
    preprocessed_data_x, mask = logit_transform(data[:, 1:-1],
                                                 params["min"], params["max"])
    preprocessed_data = np.hstack([data[:, 0:1], 
                                   preprocessed_data_x, data[:, -1:]])[mask]
    preprocessed_data[:, 1:-1] = standardize(preprocessed_data[:, 1:-1], 
                                             params["mean"], params["std"])
    return preprocessed_data