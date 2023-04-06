import numpy as np
import matplotlib.pyplot as plt

# Define the number of elements in the array
N = 12

# Define the spacing between the elements
d = 0.5

# Define the wavenumber
k = 2*np.pi


# Define the array factor
theta = np.linspace(-np.pi/2, np.pi/2, 1000)
AF = np.zeros(theta.shape, dtype=np.complex128)

for n in range(N):
    AF += 1*np.exp(1j*k*n*d*np.cos(theta))

# Plot the normalized pattern in dB vs. θ
norm = np.max(20*np.log10(np.abs(AF)))
plt.plot(np.rad2deg(theta), 20*np.log10(np.abs(AF))-norm)
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalized magnitude (dB)')
plt.title('Array pattern')
plt.ylim(-30,0)
plt.grid()


# Compute the directivity (from direct pattern integration)
phi = np.linspace(0, 2*np.pi, 361)
P = np.zeros(phi.shape, dtype=np.float64)

for p in range(phi.shape[0]):
    P[p] = np.sum(np.abs(AF*np.exp(-1j*k*d*np.cos(theta)*np.sin(phi[p]))))**2
    
directivity = 4*np.pi*np.max(P)/np.sum(P)
print('Directivity =', directivity)

# Compute the peak sidelobe level (first sidelobe) in dB
psl = np.max(20*np.log10(np.abs(AF)))-np.max(20*np.log10(np.abs(AF[theta>0])))
print('Peak sidelobe level =', psl, 'dB')

plt.show()