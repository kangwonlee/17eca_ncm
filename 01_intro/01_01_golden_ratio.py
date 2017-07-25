# golden ratio
phi = (1 + 5 ** 0.5) * 0.5
print('%g' % phi)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

ax = plt.gca()
ax.cla()

phi = (1 + 5 ** 0.5) * 0.5

# add a rectangle
rect = patches.Rectangle((0.0, 0.0), phi, 1.0, )
rect_margin = patches.Rectangle((0.0, 0.0), phi * 1.1, 1.1, fill=False, edgecolor="none")
ax.add_patch(rect)
ax.add_patch(rect_margin)
plt.axis('equal')
plt.grid(True)
plt.show()
