import numpy as np
import pandas as pd
import math
import multiprocessing
from itertools import product
from scipy.optimize import minimize

max_velocity = 15.0      # m/s
max_acceleration = 20.0  # m/s^2
num_samples = 1000

def poly_pos(t): return np.array([1, t, t**2, t**3, t**4, t**5, t**6, t**7])
def poly_vel(t): return np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6])
def poly_acc(t): return np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5])

def get_Q_matrix(T):
    Q = np.zeros((8, 8))
    for i in range(4, 8):
        for j in range(4, 8):
            power_i = i * (i-1) * (i-2) * (i-3)
            power_j = j * (j-1) * (j-2) * (j-3)
            Q[i, j] = 2 * power_i * power_j * (T**(i + j - 7)) / (i + j - 7)
    return Q

def solve_multisegment_1D(waypoints, times):
    n_segments = len(times)
    n_vars = 8 * n_segments
    Q_total = np.zeros((n_vars, n_vars))
    for i in range(n_segments): Q_total[i*8 : (i+1)*8, i*8 : (i+1)*8] = get_Q_matrix(times[i])

    A_eq, b_eq = [], []
    # Start
    A_eq.append(np.pad(poly_pos(0), (0, n_vars - 8))); b_eq.append(waypoints[0])
    A_eq.append(np.pad(poly_vel(0), (0, n_vars - 8))); b_eq.append(0)
    A_eq.append(np.pad(poly_acc(0), (0, n_vars - 8))); b_eq.append(0)

    # Continuity
    for i in range(n_segments - 1):
        T_seg = times[i]
        idx1, idx2 = i * 8, (i + 1) * 8
        row = np.zeros(n_vars); row[idx1 : idx1+8] = poly_pos(T_seg); A_eq.append(row); b_eq.append(waypoints[i+1])
        row = np.zeros(n_vars); row[idx2 : idx2+8] = poly_pos(0); A_eq.append(row); b_eq.append(waypoints[i+1])
        row = np.zeros(n_vars); row[idx1 : idx1+8] = poly_vel(T_seg); row[idx2 : idx2+8] = -poly_vel(0); A_eq.append(row); b_eq.append(0)
        row = np.zeros(n_vars); row[idx1 : idx1+8] = poly_acc(T_seg); row[idx2 : idx2+8] = -poly_acc(0); A_eq.append(row); b_eq.append(0)

    # End
    T_final = times[-1]
    idx_final = (n_segments - 1) * 8
    row = np.zeros(n_vars); row[idx_final : idx_final+8] = poly_pos(T_final); A_eq.append(row); b_eq.append(waypoints[-1])
    row = np.zeros(n_vars); row[idx_final : idx_final+8] = poly_vel(T_final); A_eq.append(row); b_eq.append(0)
    row = np.zeros(n_vars); row[idx_final : idx_final+8] = poly_acc(T_final); A_eq.append(row); b_eq.append(0)

    constraints = {'type': 'eq', 'fun': lambda c: np.dot(np.array(A_eq), c) - np.array(b_eq)}
    result = minimize(lambda c: np.dot(c.T, np.dot(Q_total, c)), np.zeros(n_vars), method='SLSQP', constraints=constraints)
    return result.x.reshape((n_segments, 8)) if result.success else None

def generate_single_sample(seed):
    np.random.seed(seed) 
    
    while True:
        # Geometry
        waypoints = [
            (np.random.uniform(0, 2), np.random.uniform(0, 2)),
            (np.random.uniform(3, 5), np.random.uniform(2, 5)),
            (np.random.uniform(6, 8), np.random.uniform(5, 8)),
            (np.random.uniform(8, 10), np.random.uniform(8, 10))
        ]
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]
        
        # Calculating inputs
        def calc_dist(p1, p2): return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        d1, d2, d3 = calc_dist(waypoints[0], waypoints[1]), calc_dist(waypoints[1], waypoints[2]), calc_dist(waypoints[2], waypoints[3])
        
        def calc_angle(p1, p2, p3):
            v1, v2 = np.array([p2[0]-p1[0], p2[1]-p1[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
            cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
            return np.arccos(cos_angle)
        
        angle1, angle2 = calc_angle(waypoints[0], waypoints[1], waypoints[2]), calc_angle(waypoints[1], waypoints[2], waypoints[3])

        # optimized grid
        time_guesses = list(product(np.arange(1.0, 3.0, 0.5), repeat=3))
        best_times = None
        min_total_time = float('inf')
        
        for times in time_guesses:
            coeffs_x = solve_multisegment_1D(wp_x, times)
            coeffs_y = solve_multisegment_1D(wp_y, times)
            
            if coeffs_x is not None and coeffs_y is not None:
                # limit check
                safe = True
                for i in range(len(times)):
                    for t in np.linspace(0, times[i], 10):
                        v_mag = math.sqrt(np.dot(poly_vel(t), coeffs_x[i])**2 + np.dot(poly_vel(t), coeffs_y[i])**2)
                        a_mag = math.sqrt(np.dot(poly_acc(t), coeffs_x[i])**2 + np.dot(poly_acc(t), coeffs_y[i])**2)
                        if v_mag > max_velocity or a_mag > max_acceleration:
                            safe = False
                            break
                    if not safe: break
                
                if safe and sum(times) < min_total_time:
                    min_total_time = sum(times)
                    best_times = times
                    
        if best_times is not None:
            return [d1, d2, d3, angle1, angle2, best_times[0], best_times[1], best_times[2]]


if __name__ == '__main__':
    print(f"Generating {num_samples} perfect samples using {multiprocessing.cpu_count()} cores...")
    
    with multiprocessing.Pool() as pool:
        # Maps the generation function across all available CPU cores
        results = pool.map(generate_single_sample, range(num_samples))
        
    df = pd.DataFrame(results, columns=['d1', 'd2', 'd3', 'angle1', 'angle2', 'T1', 'T2', 'T3'])
    df.to_csv("perfect_trajectory_dataset.csv", index=False)
    print("Generation complete! Saved to 'trajectory_dataset.csv'.")