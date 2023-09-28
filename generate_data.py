# generate data for mixed-effect model
import numpy as np
import pandas as pd

# Y = Z * omega + X * beta + e
# np.random.seed(0)
n = 500
p = 100
c = 30
X = np.random.randn(n, p) * 0.1
Z = np.random.randn(n, c)
beta = np.random.randn(p, 1)
omega = np.random.randn(c, 1) + 1.5
sigma_beta2 = 0.2
sigma_e2 = 0.01
e = np.random.randn(n, 1) * np.sqrt(sigma_e2)

y = Z @ omega + X @ beta + e
# write data to txt file
data = pd.DataFrame(np.hstack((y, Z, X)))
data.to_csv('data/fake_data.txt', sep='\t', index=False, header=True)
print('data is generated')