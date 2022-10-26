import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

# get angle (radian) between two vectors a and b (represented as arrays)
def get_angle(a, b):
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(dot_product / norm_product)

# solid angle about Point O in a tetrahedral OABC using L'Huilier's Thm
def solid_angle(theta_a, theta_b, theta_c):
    theta_s = 0.5*(theta_a + theta_b + theta_c)
    temp = math.sqrt(math.tan(0.5*theta_s) * math.tan(0.5*(theta_s - theta_a)) * \
        math.tan(0.5*(theta_s - theta_b)) * math.tan(0.5*(theta_s - theta_c)))
    return 4*math.atan(temp)

# points = np.array([[0,0,0],[2,1,1],[2,3,1],[3,1,4],[4,4,4],[3,4,0]])
points = np.array([[0,0,0],[2,1,1],[2,3,1],[3,1,4]])
delaunay_map = Delaunay(points)
print(delaunay_map.simplices)

theta_a = get_angle(np.subtract(points[2], points[0]), np.subtract(points[3], points[0]))
theta_b = get_angle(np.subtract(points[1], points[0]), np.subtract(points[3], points[0]))
theta_c = get_angle(np.subtract(points[1], points[0]), np.subtract(points[2], points[0]))
sol_ang = solid_angle(theta_a, theta_b, theta_c)
print(f"Solid angle of point 0 = {sol_ang}")

# plot two 3d subplots
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=delaunay_map.simplices)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=delaunay_map.simplices)
plt.show()