import matplotlib.pyplot as plt
import numpy as np

for i in range(100):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 1] = 255
    image[:, :, 2] = 0
    plt.imsave('data/yellow/' + str(i) + '.jpg', image)
