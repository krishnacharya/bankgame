import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_game_probability_comparison(game_results, titles=None, figsize=(15, 10)):
    """
    Plot Bank 1's probabilities over time, comparing different games side by side.
    Labels actions using gamma and tau values.
    
    Parameters:
    game_results (list): List of tuples containing (p_b1, p_b2, gammas, taus) from game.run_hedge()
    titles (list): List of titles for each game plot. If None, uses default numbering
    figsize (tuple): Figure size as (width, height)
    """
    # Validate input
    if not game_results:
        raise ValueError("No game results provided")
    
    num_games = len(game_results)
    T, num_actions = game_results[0][0].shape  # Get dimensions from first game
    
    # Calculate number of rows and columns for subplots
    num_cols = min(3, num_games)  # Max 3 columns
    num_rows = (num_games + num_cols - 1) // num_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(-1, 1) if num_cols == 1 else axes.reshape(1, -1)
    
    # Generate colors using a color map
    color_map = plt.cm.hsv
    colors = [color_map(i / num_actions) for i in range(num_actions)]
    
    # Time steps array
    time_steps = np.arange(T)
    
    # Create plots
    for idx, (p_b1, _, gammas, taus, *_) in enumerate(game_results):
        ax = axes.flatten()[idx]
        
        # Get the number of gammas and taus for the current game
        g = len(gammas)
        t = len(taus)
        
        # Plot each action's probability over time
        for action in range(num_actions):
            # Calculate the appropriate gamma and tau indices
            gamma_idx = action % g
            tau_idx = (action // g) % t  # Ensure we wrap tau_idx correctly if taus vary
            #print(action, gamma_idx, tau_idx)
            
            # Get the actual gamma and tau values
            gamma = gammas[gamma_idx]
            tau = taus[tau_idx]
            
            # Get the probability array for this action
            prob_array = p_b1[:, action]
            
            # Plot the line
            line, = ax.plot(time_steps, prob_array, 
                   color=colors[action], 
                   label=f'γ={gamma:.2f}, τ={tau:.2f}',
                   linewidth=2)
            
            # Add label if final probability is >= 0.1
            if prob_array[-1] >= 0.1:
                ax.text(time_steps[-1], prob_array[-1], 
                        f'γ={gamma:.2f}\nτ={tau:.2f}', 
                        color=line.get_color(), 
                        fontweight='bold',
                        horizontalalignment='left',
                        verticalalignment='center')
        
        # Customize each subplot
        ax.set_title(f'Game {idx+1}' if titles is None else titles[idx])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Probability')
        
        # Handle legend placement based on number of actions
        if num_actions <= 10:
            ax.legend()
        else:
            # For many actions, place legend outside
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Hide empty subplots if any
    for idx in range(len(game_results), len(axes.flatten())):
        axes.flatten()[idx].set_visible(False)
    
    plt.suptitle("Comparison of Bank 1's Action Probabilities Across Games", fontsize=14)
    plt.tight_layout()
    return fig, axes

def plot_game_probability_single(game_result, figsize=(12, 5)):
    """
    Plots the probability evolution of actions for Bank 1 and Bank 2.

    Parameters:
    - game_result (tuple): A single tuple containing (p_b1, p_b2, gammas, taus).
    - figsize (tuple): Size of the figure.
    """
    # Unpack the game result
    p_b1, p_b2, gammas, taus = game_result
    
    # Ensure gammas and taus have the same length
    n = len(gammas)
    assert len(taus) == n, "Number of gammas and taus must be equal"

    T, num_actions = p_b1.shape  # Number of time steps and actions
    assert num_actions == n * n, "Mismatch between number of actions and (tau, gamma) pairs"

    time_steps = np.arange(T)  # Time steps
    color_map = plt.cm.hsv
    colors = [color_map(i / num_actions) for i in range(num_actions)]  # Assign distinct colors

    fig, axes = plt.subplots(1, 2, figsize=figsize)  # Two subplots for Bank 1 & Bank 2

    for i, (bank_pw, title, ax) in enumerate(zip([p_b1, p_b2], ["Bank 1", "Bank 2"], axes)):
        for action in range(num_actions):
            tau_idx = action // n
            gamma_idx = action % n
            tau = taus[tau_idx]
            gamma = gammas[gamma_idx]

            # Plot probability evolution
            line, = ax.plot(time_steps, bank_pw[:, action], 
                            color=colors[action], 
                            label=f'(τ={tau:.2f}, γ={gamma:.2f})',
                            linewidth=2)
            
#             # Annotate actions that have significant probability at the final time step
#             if bank_pw[-1, action] >= 0.1:
#                 ax.text(time_steps[-1], bank_pw[-1, action], 
#                         f'τ={tau:.2f}\nγ={gamma:.2f}', 
#                         color=line.get_color(), 
#                         fontweight='bold',
#                         horizontalalignment='left',
#                         verticalalignment='center')

        # Customize plot
        ax.set_title(f'{title} Action Probabilities Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()