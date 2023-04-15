import numpy as np
import matplotlib.pyplot as plt

# Define the number of elements in the array
N = 12

# Define the spacing between the elements
d = 0.5

# Define the wavenumber
k = 2*np.pi

# Beta value for array factor
B = 0 # Boadside array
#B = np.pi/2 # Standard ULA (Main beam at 0 degrees)

# Define the array factor

theta = np.linspace(-np.pi/2, np.pi/2, 1000)
#theta = np.linspace(0, np.pi, 1000)

AF = np.zeros(theta.shape, dtype=np.complex128)

#ArrayWeights= np.array([-5.790, -2.050, 0.485, 1.177, 0.161, -1.00, 0, 1.00, -0.161, -1.177, -0.485, 2.050, 5.790])
ArrayWeights= np.array([-1.00, 0.161, 1.177, 0.485, -2.050, -5.790,0, 5.790, 2.050, -0.485, -1.177, -0.161, 1.00])

for n in [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]:
    AF += ArrayWeights[n+6] * np.exp(1j*k*n*d*np.cos(theta + B) )
    #AF += np.exp(1j*k*n*d*np.cos(theta + B) )

#Plot the normalized pattern in dB vs. θ
norm = np.max(20*np.log10(np.abs(AF)))
plt.plot(np.rad2deg(theta), 20*np.log10(np.abs(AF))-norm)
#plt.plot(np.rad2deg(theta), 20*(np.abs(AF)))
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalized magnitude (dB)')
plt.title('Array pattern')
plt.ylim(-30,0)
plt.grid()


# Compute the directivity (from direct pattern integration)


directivity = 2*N*(d)
directivity = 10*np.log10(directivity)
print('Directivity =', directivity)

# Compute the peak sidelobe level (first sidelobe) in dB
psl = np.max(20*np.log10(np.abs(AF)))-np.max(20*np.log10(np.abs(AF[theta>0])))
print('Peak sidelobe level =', psl, 'dB')

plt.show()