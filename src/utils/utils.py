import json
import decimal
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, (complex, np.complexfloating)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

ctx = decimal.Context()
ctx.prec = 20

# taken from https://stackoverflow.com/questions/38847690
def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def str_encode_value(val:float, n_digit=None, formatted=True):
    if n_digit is not None:
        val_str = '{{:.{}f}}'.format(n_digit).format(val)
    else:
        val_str = float_to_str(val)
    # edge case of negative zero
    if val_str == '-0.0':
        val_str = '0p0'
    
    if formatted:
        val_str = val_str.replace('.', 'p').replace('-', 'n')
    return val_str
