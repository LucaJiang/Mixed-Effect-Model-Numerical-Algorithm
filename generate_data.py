# generate data for mixed-effect model
import numpy as np
import pandas as pd

# Y = Z * omega + X * beta + e
np.random.seed(0)
n = 200
p = 1200
c = 30
X = np.random.randn(n, p) * 0.1
Z = np.random.randn(n, c)
omega = np.random.randn(c, 1) + 2
sigma_beta2 = 2
sigma_e2 = 0.01
beta = np.random.randn(p, 1) * np.sqrt(sigma_beta2)
e = np.random.randn(n, 1) * np.sqrt(sigma_e2)

y = Z @ omega + X @ beta + e
# write data to txt file
data = pd.DataFrame(np.hstack((y, Z, X)))
data.to_csv('data/fake_data.txt', sep='\t', index=False, header=True)
print('data is generated')