import numpy as np
from scipy import stats

# http://benalexkeen.com/correlation-in-python/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
# https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/

a = np.array([0, 0, 0, 1, 1, 1, 1])
b = np.arange(7)

coefficient, p_value = stats.pearsonr(a, b)
