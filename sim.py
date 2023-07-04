import numpy as np
import jPCA
import matplotlib.pyplot as plt
from jPCA.util import load_churchland_data, plot_projections

# Generate 3D data
n = 1000
t = np.linspace(0, 10, n)
extra_dims = 20
cycle = t / (1 * np.pi) 
# print(t)
datas = []

for _ in range(50):
    x = np.sin(cycle) + np.random.normal(0, 0.1, n)
    y = np.cos(cycle) + np.random.normal(0, 0.1, n)
    z = t + np.random.normal(0.1, 0.01, n)
    tmp = np.hstack(
        (np.array([x, y, z]).T, 
        np.random.multivariate_normal(
            np.zeros(extra_dims)*0.01,np.eye(extra_dims),t.shape[0]
        ))
    )
    # tmp = np.array([x, y, z]).T
    # tmp = np.hstack((
    #     np.array([x, y, z]).T, 
    #     np.zeros((t.shape[0],extra_dims))  
    # ))
    

    datas.append(tmp)
times = list(t)

print("\nInput:")
print(f'len(datas) = {len(datas)}')
print(f'datas[0].shape = {datas[0].shape}')
print(f'len(times) = {len(times)}')
# exit()

# Run jPCA  
jpca = jPCA.JPCA(num_jpcs=4)

(projected, 
 full_data_var,
 pca_var_capt,
 jpca_var_capt) = jpca.fit(
    datas, times=times, tstart=t[0], tend=t[-2]
)

print("\nOutput:")
# print(np.array_equal(projected[0], np.zeros_like(projected[0])))
print(f'len(projected) = {len(projected)}')
print(f'projected[0].shape = {projected[0].shape}')

plt.plot(x, y)
plt.plot(datas[0][:,0], datas[0][:,1])
# plt.plot(projected[2][:50,0], projected[0][:50,1])
plt.legend()
plt.show()

exit()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_projections(projected[:1], axis=axes[0], x_idx=0, y_idx=1)
plot_projections(projected[:2], axis=axes[1], x_idx=2, y_idx=3)

axes[0].set_title("jPCA Plane 1")
axes[1].set_title("jPCA Plane 2")
plt.tight_layout()
# axes[0].set_xlim(-1, 1)
# axes[0].set_ylim = (-1, 1)
plt.show()

exit()
# Load publicly available data from Mark Churchland's group
path = "exampleData.mat"
datas, times = load_churchland_data(path)

print(type(times))
print(datas[0].shape)

# Create a jPCA object
jpca = jPCA.JPCA(num_jpcs=2)

# datas, times = np.array(datas), np.array(times)
datas, times = datas[:2], times[:]

(projected, 
 full_data_var,
 pca_var_capt,
 jpca_var_capt) = jpca.fit(datas, times=times, tstart=-50, tend=150)

