import numpy as np
import matplotlib.pyplot as plt

N = 20 # Number of elements in the array
d = 0.5 # Element spacing in wavelengths
theta = np.linspace(-np.pi/2, np.pi/2, 1000) # Theta values for plotting

# Set up the array weights with arbitrary amplitude and phase
w = np.random.rand(N) * np.exp(1j*np.random.rand(N)*2*np.pi)

# Compute the array factor using complex exponentials
af = np.zeros_like(theta, dtype=np.complex64)
for i in range(N):
    af += w[i] * np.exp(1j*2*np.pi*i*d*np.sin(theta))

# Compute the normalized array pattern in dB
ap = 20*np.log10(np.abs(af)/np.max(np.abs(af)))

# Plot the normalized array pattern in dB vs. theta
plt.plot(theta, ap)
plt.xlabel('Angle (radians)')
plt.ylabel('Normalized Array Pattern (dB)')
plt.title('Linear Array Pattern')
plt.grid(True)

# Compute the directivity and peak sidelobe level
D = 4*np.pi/np.sum(np.abs(af)**2)
psl = np.max(ap) - np.max(ap[np.abs(theta) > np.pi/N])

print('Directivity: {:.2f}'.format(D))
print('Peak Sidelobe Level: {:.2f} dB'.format(psl))
