import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

waypoints = np.array([[0, 0, 1], [3.0, 2.0, 1.5], [6.0, -1.0, 2.0], [9.0, 1.0, 1.0]])
segments = np.diff(waypoints, axis=0) #vector
distance = np.linalg.norm(segments, axis=1) #segment distance
speed = 1.0 # m/s
time = distance / speed # time for each segment
#how many time have passed for each segments
total_distance = np.sum(distance)
heuristic = speed * (distance / total_distance) # heuristic time for each segment
for i, t_val in enumerate(heuristic):
    print(f"Segment {i+1} T_{i+1}: {t_val:3f}s")

#cumulative time break for the waypoints
cumulative_time = np.insert(np.cumsum(heuristic), 0, 0.0)

#Mapping the spatial coordinates against cumulative time with cubic spline
fine_time_step = np.linspace(0, speed, 200)
cs_x = CubicSpline(cumulative_time, waypoints[:, 0], bc_type='natural')
cs_y = CubicSpline(cumulative_time, waypoints[:, 1], bc_type='natural')
cs_z = CubicSpline(cumulative_time, waypoints[:, 2], bc_type='natural')
position_trajectory = np.vstack((cs_x(fine_time_step), cs_y(fine_time_step), cs_z(fine_time_step))).T
velocity_trajectory = np.vstack((cs_x(fine_time_step, 1), cs_y(fine_time_step, 1), cs_z(fine_time_step, 1))).T
acceleration_trajectory = np.vstack((cs_x(fine_time_step, 2), cs_y(fine_time_step, 2), cs_z(fine_time_step, 2))).T

#Visualizing the 3d trajectory and dynamics with Matplotlib.py
fig = plt.figure(figsize=(12, 8))
# 3D Trajectory Plot
ax1 = fig.add_subplot(2, 2, (1, 2), projection='3d')
ax1.plot(position_trajectory[:, 0], position_trajectory[:, 1], position_trajectory[:, 2],'b-', linewidth = 2.5, label='Trajectory (Spline)')
ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color = 'red', s=100, label='Waypoints(P0-P3)')
ax1.set_title('3D Multi-Segment trajectory path', fontsize = 14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
ax1.grid(True)

#Velocity Plot
ax2 = fig.add_subplot(2, 2, 3)
speeds = np.linalg.norm(velocity_trajectory, axis=1)
ax2.plot(fine_time_step, speeds, 'g-', linewidth = 2.5)
ax2.set_title('Velocity MagnitudeProfile', fontsize = 14)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Speed/Velocity (m/s)')
ax2.grid(True)

#Acceleration/ Thrust Proxy Plot
ax3 = fig.add_subplot(2, 2, 4)
accelerations = np.linalg.norm(acceleration_trajectory, axis=1)
ax3.plot(fine_time_step, accelerations, 'r-', linewidth = 2.5)
ax3.set_title('Acceleration Magnitude Profile', fontsize = 14)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.grid(True)

plt.tight_layout()
plt.show()