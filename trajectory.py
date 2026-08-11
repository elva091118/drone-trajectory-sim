import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def poly_pos(t):
    return np.array([1, t, t**2, t**3, t**4, t**5, t**6, t**7])

def poly_vel(t):
    return np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6])

def poly_acc(t):
    return np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5])

def get_Q_matrix(T):
    """Generates the 8x8 Hessian (Snap cost) for a single segment of time T."""
    Q = np.zeros((8, 8))
    for i in range(4, 8):
        for j in range(4, 8):
            power_i = i * (i-1) * (i-2) * (i-3)
            power_j = j * (j-1) * (j-2) * (j-3)
            Q[i, j] = 2 * power_i * power_j * (T**(i + j - 7)) / (i + j - 7)
    return Q

def solve_multisegment_1D(waypoints, times):
    """Solves the 1D minimum snap trajectory across multiple segments."""
    n_segments = len(times)
    n_vars = 8 * n_segments
    
    Q_total = np.zeros((n_vars, n_vars))
    for i in range(n_segments):
        Q_total[i*8 : (i+1)*8, i*8 : (i+1)*8] = get_Q_matrix(times[i])

    def objective(c):
        return np.dot(c.T, np.dot(Q_total, c))

    A_eq = []
    b_eq = []

    A_eq.append(np.pad(poly_pos(0), (0, n_vars - 8)))
    b_eq.append(waypoints[0])
    A_eq.append(np.pad(poly_vel(0), (0, n_vars - 8)))
    b_eq.append(0) 
    A_eq.append(np.pad(poly_acc(0), (0, n_vars - 8)))
    b_eq.append(0) 

    for i in range(n_segments - 1):
        T_seg = times[i]
        idx1 = i * 8
        idx2 = (i + 1) * 8
        
        row = np.zeros(n_vars)
        row[idx1 : idx1+8] = poly_pos(T_seg)
        A_eq.append(row)
        b_eq.append(waypoints[i+1])
        
        row = np.zeros(n_vars)
        row[idx2 : idx2+8] = poly_pos(0)
        A_eq.append(row)
        b_eq.append(waypoints[i+1])
        
        # Velocity matches
        row = np.zeros(n_vars)
        row[idx1 : idx1+8] = poly_vel(T_seg)
        row[idx2 : idx2+8] = -poly_vel(0)
        A_eq.append(row)
        b_eq.append(0)
        
        # Continuity
        row = np.zeros(n_vars)
        row[idx1 : idx1+8] = poly_acc(T_seg)
        row[idx2 : idx2+8] = -poly_acc(0)
        A_eq.append(row)
        b_eq.append(0)

    T_final = times[-1]
    idx_final = (n_segments - 1) * 8
    
    row = np.zeros(n_vars)
    row[idx_final : idx_final+8] = poly_pos(T_final)
    A_eq.append(row)
    b_eq.append(waypoints[-1])
    
    row = np.zeros(n_vars)
    row[idx_final : idx_final+8] = poly_vel(T_final)
    A_eq.append(row)
    b_eq.append(0) 
    
    row = np.zeros(n_vars)
    row[idx_final : idx_final+8] = poly_acc(T_final)
    A_eq.append(row)
    b_eq.append(0)

    constraints = {'type': 'eq', 'fun': lambda c: np.dot(np.array(A_eq), c) - np.array(b_eq)}
    c_guess = np.zeros(n_vars)
    
    result = minimize(objective, c_guess, method='SLSQP', constraints=constraints)
    
    if result.success:
        return result.x.reshape((n_segments, 8))
    else:
        print("Optimization Failed.")
        return None


if __name__ == "__main__":
    wp_x = [0.0, 2.2, 5.3, 6.0]
    wp_y = [0.0, 3.1, 1.8, 4.0]
    times = [1.5, 2.0, 1.2]

    # X and Y independently
    coeffs_x = solve_multisegment_1D(wp_x, times)
    coeffs_y = solve_multisegment_1D(wp_y, times)

    plt.figure(figsize=(8, 6))
    
    for i in range(len(times)):
        t_vals = np.linspace(0, times[i], 50)
        x_vals = [np.dot(poly_pos(t), coeffs_x[i]) for t in t_vals]
        y_vals = [np.dot(poly_pos(t), coeffs_y[i]) for t in t_vals]
        plt.plot(x_vals, y_vals, label=f'Segment {i+1} ({times[i]}s)')

    plt.scatter(wp_x, wp_y, color='red', s=100, zorder=5, label='Waypoints')
    plt.title('Milestone 1: 2D Minimum Snap Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()