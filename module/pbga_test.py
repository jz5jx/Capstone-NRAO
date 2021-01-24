import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from pbga import PBGA

if __name__ == "__main__":
    path = "C:/Users/pavan/Projects/Capstone/Tobin_Data/"
    file = fits.open(path + "HH270VLA1_cont_robust0.5.pbcor.fits")
    image = file[0].data[0, 0, :, :]
    std = np.std(image)
    image[image < std * 8.5] = 0

    pbga = PBGA(buffer_size=10, group_size=40)
    pbga.run(image)

    group_ranges = pbga.group_ranges
    group_data = pbga.group_data
    group_stats = pbga.group_stats

    for data_ in group_data:
        plt.imshow(data_['IMAGE'], origin='lower')
        plt.show()
