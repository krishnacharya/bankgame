import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
from PIL import Image

def plot_game_probability_single(game_result, figsize=(12, 5), multi = False):
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
            
        # Customize plot
        ax.set_title(f'{title} Action Probabilities Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Convert buffer to PIL Image
    image = Image.open(buf)
    
    # Display the plot
    if not multi:
        plt.show()
        plt.close(fig)
    else:
        return image
    
def plot_multiple_game_probabilities(game_results, titles = None, figsize=(12, 5), display=True):
    """
    Plots multiple game result probability evolutions for Bank 1 and Bank 2.

    Parameters:
    - game_results (list of tuples): List containing tuples of game results.
    - figsize (tuple): Size of the figure.
    - multi (bool): Whether to return the image buffer or display it.
    """
    num_games = len(game_results)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_games, 2, figsize=figsize, sharex=True, sharey=True)

    # If there's only one game result, axes will be a 2D array with one row
    if num_games == 1:
        axes = np.expand_dims(axes, axis=0)

    for game_idx, game_result in enumerate(game_results):
        # Unpack the game result
        p_b1, p_b2, gammas, taus = game_result
        n = len(gammas)
        assert len(taus) == n, "Number of gammas and taus must be equal"

        T, num_actions = p_b1.shape  # Number of time steps and actions
        assert num_actions == n * n, "Mismatch between number of actions and (tau, gamma) pairs"

        time_steps = np.arange(T)  # Time steps
        color_map = plt.cm.hsv
        colors = [color_map(i / num_actions) for i in range(num_actions)]  # Assign distinct colors

        for i, (bank_pw, title, ax) in enumerate(zip([p_b1, p_b2], ["Bank 1", "Bank 2"], axes[game_idx])):
            for action in range(num_actions):
                tau_idx = action // n
                gamma_idx = action % n
                tau = taus[tau_idx]
                gamma = gammas[gamma_idx]

                # Plot probability evolution for this game result
                ax.plot(time_steps, bank_pw[:, action], 
                        color=colors[action], 
                        label=f'(τ={tau:.2f}, γ={gamma:.2f})',
                        linewidth=2)
            
            # Customize plot for each subplot
            if not titles:
                ax.set_title(f'{title} Action Probabilities (Game {game_idx+1})')
            else:
                ax.set_title(f'{title} Action Probabilities ({titles[game_idx]})')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Convert buffer to PIL Image
    image = Image.open(buf)
    
    # Display the plot
    if display:
        plt.show()
        plt.close(fig)
        return image
    else:
        return image