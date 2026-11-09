import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 100

x_linear, step_size = np.linspace(start=0,stop=10,num=15,retstep=True)
print("Generated Array ")
print(f"Calculated step size: {step_size:.4f}")

plt.figure(figsize= (9,2.5))
plt.scatter(x_linear,np.zeros_like(x_linear),color='blue', s=60, zorder=3,label=f'Linear Samples (h={step_size:.2f})')

plt.vlines(x_linear, ymin=-0.1, ymax=0.1, colors='#1f77b4', s=60,zorder=3,)