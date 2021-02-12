import numpy as np
import matplotlib.pyplot as plt

labels = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
scores = [0.1, 0.9, 0.5, 0.4, 0.01, 0.65, 0.95, 0.03, 0.6, 0.2]

# 1. Model Calibration.

from sklearn.calibration import calibration_curve
true_prod, pred_prob = 	calibration_curve(labels, scores, n_bins=10)

plt.title('Title')
plt.xlabel('Horizontal Axis')
plt.ylabel('Vertical Axis')
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated.")
plt.plot(pred_prob, true_prod, "s-", label="Model.")
plt.show()
plt.clf()

# 2. Reliability.

vals, bins = np.histogram(scores, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

plt.title('Title')
plt.xlabel('Horizontal Axis')
plt.ylabel('Vertical Axis')
plt.hist(vals, bins=bins)
plt.show()
plt.clf()