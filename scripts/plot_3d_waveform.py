import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    # Load the FD data
    data_path = 'outputs/pinn/zerilli_l2/zerilli_l2_fd.npz'
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    x = data['x']
    t = data['t']
    phi = data['phi']
    
    print(f"Data loaded. x shape: {x.shape}, t shape: {t.shape}, phi shape: {phi.shape}")

    # Create a meshgrid for plotting
    X, T = np.meshgrid(x, t)

    # Create the figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample the data for plotting so it doesn't crash or take forever
    # We want roughly a 200x200 grid for a smooth but manageable 3D surface
    stride_t = max(1, len(t) // 200)
    stride_x = max(1, len(x) // 200)

    print(f"Plotting surface with strides: t={stride_t}, x={stride_x}...")
    surf = ax.plot_surface(X[::stride_t, ::stride_x], 
                           T[::stride_t, ::stride_x], 
                           phi[::stride_t, ::stride_x], 
                           cmap='plasma', # Plasma is great for highlighting peaks and valleys
                           edgecolor='none',
                           alpha=0.9)

    # Add labels and title
    ax.set_xlabel('Space (x*)', fontsize=12, labelpad=10)
    ax.set_ylabel('Time (t)', fontsize=12, labelpad=10)
    ax.set_zlabel('Wave Amplitude (u)', fontsize=12, labelpad=10)
    ax.set_title('The 3D Landscape of a Black Hole Ringdown (FD Solution)', fontsize=16, pad=20)

    # Adjust the viewing angle to clearly see the time evolution
    # elev=30 (look slightly down), azim=-50 (look from the side to see time progress)
    ax.view_init(elev=35, azim=-55)

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Amplitude', pad=0.1)

    # Save the plot
    out_path = 'outputs/pinn/zerilli_l2/waveform_3d.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved 3D plot to {out_path}")

if __name__ == "__main__":
    main()
