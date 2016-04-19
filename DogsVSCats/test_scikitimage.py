from main_resnet import load_dataset1
data = load_dataset1(10,(200,200),True)
import numpy as np

import matplotlib.pyplot as plt

ei = data[0].get_epoch_iterator()
d= next(ei)

print(d)

#plt.imshow(np.rollaxis(d[0][1],0,3))
#plt.show()
