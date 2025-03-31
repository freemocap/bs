import numpy as np
from matplotlib import pyplot as plt

def plot_optical_flow_histograms(raw_image: np.ndarray, flow_image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    x_flow = flow[:, 0]
    y_flow = flow[:, 1]
    
    x_flow_range = flow_image.shape[1] / 2
    y_flow_range = flow_image.shape[0] / 2

    xlim = 25
    ylim = (flow_image.shape[0] + flow_image.shape[1]) * 0.9

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]

    ax0.imshow(flow_image)
    
    # Plot x-direction histogram
    ax1.hist(x_flow.ravel(), bins=50, range=(-x_flow_range, x_flow_range),
             color='blue', alpha=0.7)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_title('Horizontal Flow Distribution')
    ax1.set_xlabel('Flow magnitude (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_ylim(0, ylim)
    ax1.set_xlim(-xlim, xlim)
    
    # Plot y-direction histogram
    ax2.hist(y_flow.ravel(), bins=50, range=(-y_flow_range, y_flow_range),
             color='green', alpha=0.7)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_title('Vertical Flow Distribution')
    ax2.set_xlabel('Flow magnitude (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_ylim(0, ylim)
    ax2.set_xlim(-xlim, xlim)

    ax3.imshow(raw_image)
    
    plt.tight_layout()
    fig.canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi() 
    image_from_plot = np.fromstring(fig.canvas.tostring_argb(), dtype='uint8').reshape(int(height), int(width), 4)

    plt.close(fig)

    return image_from_plot[:, :, 1:]
