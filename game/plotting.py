import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_game_probability_comparison(game_results, titles=None, figsize=(15, 10)): # TODO fix indexing here too now gamma is the faster changing index
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
        
        # Calculate n for pair indexing
        n = len(taus)
        
        # Plot each action's probability over time
        for action in range(num_actions):
            # Convert action index to gamma and tau indices
            gamma_idx = action // n
            tau_idx = action % n
            
            # Get the actual gamma and tau values
            gamma = gammas[gamma_idx]
            tau = taus[tau_idx]
            
            ax.plot(time_steps, p_b1[:, action], 
                   color=colors[action], 
                   label=f'γ={gamma:.2f}, τ={tau:.2f}',
                   linewidth=2)
        
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