import numpy as np
from matplotlib import pyplot as plt

def plot_optical_flow_histograms(raw_image: np.ndarray, flow_image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    x_flow = flow[:, 0]
    y_flow = flow[:, 1]
    
    # Automatically determine histogram ranges
    # Adding 10% padding to ensure outliers fit within bounds
    x_range = flow_image.shape[1]
    y_range = flow_image.shape[0]
    
    print(f"x-flow range: [{x_flow.min():.2f}, {x_flow.max():.2f}]")
    print(f"y-flow range: [{y_flow.min():.2f}, {y_flow.max():.2f}]")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]

    ax0.imshow(flow_image)
    
    # Plot x-direction histogram
    ax1.hist(x_flow.ravel(), bins=50, range=(-x_range, x_range),
             color='blue', alpha=0.7)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_title('Horizontal Flow Distribution')
    ax1.set_xlabel('Flow magnitude (pixels)')
    ax1.set_ylabel('Frequency')
    
    # Plot y-direction histogram
    ax2.hist(y_flow.ravel(), bins=50, range=(-y_range, y_range),
             color='green', alpha=0.7)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_title('Vertical Flow Distribution')
    ax2.set_xlabel('Flow magnitude (pixels)')
    ax2.set_ylabel('Frequency')

    ax3.imshow(raw_image)
    
    plt.tight_layout()
    fig.canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi() 
    image_from_plot = np.fromstring(fig.canvas.tostring_argb(), dtype='uint8').reshape(int(height), int(width), 4)
    # image_from_plot = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # canvas_width, canvas_height = fig.canvas.get_width_height()
    # expected_size = canvas_width * canvas_height * 4

    # if image_from_plot.size == expected_size:
    #     image_from_plot = image_from_plot.reshape((canvas_height, canvas_width, 4))
    #     image_from_plot = image_from_plot[:, :, 1:]  # Convert ARGB to RGB by dropping the alpha channel
    # else:
    #     print(f"Mismatched size between buffer and expected dimensions. Expected: {expected_size}, Actual: {image_from_plot.size}")
    #     raise ValueError("Mismatched size between buffer and expected dimensions.")
    # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return image_from_plot[:, :, 1:]
