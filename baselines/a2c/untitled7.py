# import necessary module
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# load data from file
# you can replace this using with open
data1 = np.loadtxt("./stereo/CameraTrajectoryNew2000.txt")
first_2000 = data1[:, 3]
second_2000 = data1[:, 7]
third_2000 = data1[:, 11]

# print to check data
print first_2000
print second_2000
print third_2000

# new a figure and set it into 3d
fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
ax.set_title("3D_Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# draw the figure, the color is r = read
figure = ax.plot(first_2000, second_2000, third_2000, c='r')

plt.show()



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
 
 
fig = plt.figure()
ax = fig.gca(projection='3d')
b=range(84)
a=range(84)
a, b = np.meshgrid(a,b)

c=np.ones((84,84))

for i in range(84):
    for j in range(84):
        c[i,j]=oj[i*84+j]
        if c[i,j]==0:
            c[i,j]=-1
surf = ax.plot_surface(a, b, c, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

 
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
 
plt.show()















from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
 
 


















fig = plt.figure()
ax = fig.gca(projection='3d')
 
# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
 
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
 
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
 
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
 
plt.show()