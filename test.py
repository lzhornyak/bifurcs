import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, initial_state, t0, t1, dt, param):
    steps = int((t1 - t0) / dt)
    t = t0
    state = initial_state

    states = []
    for _ in range(steps):
        state += dt * f(state, param)
        t += dt
        states.append(state)

    return np.array(states)

def plot_continuous_bifurcation_diagram(f, param_range, initial_state, t0, t1, dt, plot_duration):
    param_min, param_max = param_range
    param_values = np.linspace(param_min, param_max, 1000)
    bifurcation_data = []

    for param in param_values:
        states = euler_method(f, initial_state, t0, t1, dt, param)
        # Collect only the last 'plot_duration' states
        if len(states) >= plot_duration:
            sampled_states = states[-plot_duration:]
        else:
            # If not enough states, repeat the last state to fill the gap
            sampled_states = np.full(plot_duration, states[-1])
        bifurcation_data.append(np.column_stack((np.full(plot_duration, param), sampled_states)))

    bifurcation_data = np.concatenate(bifurcation_data)

    # Plot the diagram
    plt.plot(bifurcation_data[:, 0], bifurcation_data[:, 1], ',k', alpha=.25)
    plt.title("Bifurcation diagram (Continuous Time)")
    plt.xlabel("Parameter")
    plt.ylabel("State")
    plt.show()

# Example usage with a simple differential equation
def example_diff_eq(state, r):
    return r * state + state ** 3 - state ** 5

plot_continuous_bifurcation_diagram(example_diff_eq, (-3, 3), 0.1, 0, 100, 0.01, 1000)
