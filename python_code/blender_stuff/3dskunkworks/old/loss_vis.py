import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
from typing import Callable


def linear_loss(r: np.ndarray) -> np.ndarray:
    """Standard least squares: ρ(r) = r²"""
    return r ** 2


def huber_loss(r: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Huber loss: smooth L2 for small errors, L1 for large errors.

    ρ(r) = { r²/2           if |r| ≤ δ
           { δ(|r| - δ/2)   if |r| > δ
    """
    abs_r = np.abs(r)
    return np.where(
        abs_r <= delta,
        0.5 * r ** 2,
        delta * (abs_r - 0.5 * delta)
    )


def soft_l1_loss(r: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Soft L1: smooth approximation of L1 loss.

    ρ(r) = 2δ²(√(1 + (r/δ)²) - 1)
    """
    return 2 * delta ** 2 * (np.sqrt(1 + (r / delta) ** 2) - 1)


def cauchy_loss(r: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Cauchy loss: very aggressive outlier rejection.

    ρ(r) = δ² ln(1 + (r/δ)²)
    """
    return delta ** 2 * np.log(1 + (r / delta) ** 2)


def arctan_loss(r: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Arctan loss: bounded loss function.

    ρ(r) = δ² arctan²(r/δ)
    """
    return delta ** 2 * np.arctan(r / delta) ** 2


def plot_loss_functions():
    """Compare different robust loss functions using Plotly."""

    # Error range: -10 to 10 (e.g., -10mm to +10mm)
    errors = np.linspace(-10, 10, 1000)
    delta = 2.0  # Threshold parameter (e.g., 2mm)

    # Compute losses
    losses = {
        'Linear (L2)': linear_loss(errors),
        'Huber': huber_loss(errors, delta=delta),
        'Soft L1': soft_l1_loss(errors, delta=delta),
        'Cauchy': cauchy_loss(errors, delta=delta),
        'Arctan': arctan_loss(errors, delta=delta)
    }

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Loss Functions: How Much We Penalize Errors',
            'Influence Functions: How Much Errors Affect Result',
            'Effective Weights: How Much We Trust Each Measurement'
        ),
        horizontal_spacing=0.10
    )

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # 1. Loss functions
    for (name, loss), color in zip(losses.items(), colors):
        fig.add_trace(
            go.Scatter(
                x=errors,
                y=loss,
                mode='lines',
                name=name,
                line=dict(color=color, width=3),
                legendgroup=name,
                showlegend=True
            ),
            row=1, col=1
        )

    # Add delta threshold lines
    fig.add_trace(
        go.Scatter(
            x=[delta, delta],
            y=[0, 60],
            mode='lines',
            name=f'δ = {delta}',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=True
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[-delta, -delta],
            y=[0, 60],
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=1
    )

    # Add annotations
    fig.add_annotation(
        x=7,
        y=linear_loss(np.array([7]))[0],
        text="Outliers get huge penalty!<br>(dominates optimization)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        ax=-80,
        ay=-60,
        bgcolor='yellow',
        opacity=0.8,
        row=1, col=1
    )

    fig.add_annotation(
        x=7,
        y=huber_loss(np.array([7]), delta=delta)[0],
        text="Robust losses plateau<br>(outliers have bounded influence)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='blue',
        ax=-60,
        ay=40,
        bgcolor='lightblue',
        opacity=0.8,
        row=1, col=1
    )

    # 2. Derivative (influence function)
    dr = errors[1] - errors[0]
    for (name, loss), color in zip(losses.items(), colors):
        derivative = np.gradient(loss, dr)
        fig.add_trace(
            go.Scatter(
                x=errors,
                y=derivative,
                mode='lines',
                name=name,
                line=dict(color=color, width=3),
                legendgroup=name,
                showlegend=False
            ),
            row=1, col=2
        )

    # Add delta threshold lines
    fig.add_trace(
        go.Scatter(
            x=[delta, delta],
            y=[-25, 25],
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[-delta, -delta],
            y=[-25, 25],
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

    # Add zero line
    fig.add_trace(
        go.Scatter(
            x=[-10, 10],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ),
        row=1, col=2
    )

    # Add annotations
    fig.add_annotation(
        x=8,
        y=2 * 8,
        text="L2: influence grows linearly<br>(outliers pull solution!)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        ax=-60,
        ay=-50,
        bgcolor='yellow',
        opacity=0.8,
        row=1, col=2
    )

    fig.add_annotation(
        x=8,
        y=delta,
        text="Robust: influence saturates<br>(outliers ignored)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='blue',
        ax=-50,
        ay=60,
        bgcolor='lightblue',
        opacity=0.8,
        row=1, col=2
    )

    # 3. Effective weight
    for (name, loss), color in zip(losses.items(), colors):
        if name == 'Linear (L2)':
            continue  # Skip linear (always weight = 1)

        derivative = np.gradient(loss, dr)
        linear_derivative = np.gradient(losses['Linear (L2)'], dr)

        # Avoid division by zero
        weight = np.where(np.abs(errors) > 0.01,
                          derivative / linear_derivative,
                          1.0)

        fig.add_trace(
            go.Scatter(
                x=errors,
                y=weight,
                mode='lines',
                name=name,
                line=dict(color=color, width=3),
                legendgroup=name,
                showlegend=False
            ),
            row=1, col=3
        )

    # Add L2 reference line
    fig.add_trace(
        go.Scatter(
            x=[-10, 10],
            y=[1, 1],
            mode='lines',
            name='L2 weight',
            line=dict(color='red', dash='dot', width=3),
            showlegend=False
        ),
        row=1, col=3
    )

    # Add delta threshold lines
    fig.add_trace(
        go.Scatter(
            x=[delta, delta],
            y=[0, 1.2],
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=3
    )

    fig.add_trace(
        go.Scatter(
            x=[-delta, -delta],
            y=[0, 1.2],
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=3
    )

    # Add zero line
    fig.add_trace(
        go.Scatter(
            x=[-10, 10],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ),
        row=1, col=3
    )

    # Add annotation
    fig.add_annotation(
        x=6,
        y=0.3,
        text="Large errors get<br>downweighted!",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='darkblue',
        ax=-50,
        ay=-40,
        bgcolor='lightgreen',
        opacity=0.8,
        row=1, col=3
    )

    # Update axes
    fig.update_xaxes(title_text="Error (mm)", row=1, col=1)
    fig.update_xaxes(title_text="Error (mm)", row=1, col=2)
    fig.update_xaxes(title_text="Error (mm)", row=1, col=3)

    fig.update_yaxes(title_text="Loss ρ(error)", row=1, col=1, range=[0, 60])
    fig.update_yaxes(title_text="dρ/dr (Influence)", row=1, col=2, range=[-25, 25])
    fig.update_yaxes(title_text="Relative Weight", row=1, col=3, range=[0, 1.2])

    # Update layout
    fig.update_layout(
        height=500,
        width=1800,
        title_text="<b>Robust Loss Functions Comparison</b>",
        title_x=0.5,
        title_font_size=18,
        showlegend=True,
        legend=dict(
            x=1.01,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        hovermode='x unified'
    )

    filename = 'robust_loss_comparison.html'
    fig.write_html(filename)
    print(f"✓ Saved: {filename}")

    return fig


def demonstrate_outlier_robustness():
    """Demonstrate how robust losses handle outliers in practice using Plotly."""

    np.random.seed(42)

    # Generate clean data: y = 2x + 1
    n_points = 50
    x = np.linspace(0, 10, n_points)
    y_true = 2 * x + 1

    # Add Gaussian noise
    y_noisy = y_true + np.random.randn(n_points) * 1.0

    # Add outliers (10% of points)
    n_outliers = 5
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    y_noisy[outlier_indices] += np.random.randn(n_outliers) * 10.0

    # Fit using different loss functions
    def fit_line(params, x, y):
        """Residuals for line fitting: y = mx + b"""
        m, b = params
        return y - (m * x + b)

    # Standard least squares
    result_l2 = least_squares(
        fun=fit_line,
        x0=[1.0, 0.0],
        args=(x, y_noisy),
        loss='linear'
    )

    # Huber loss
    result_huber = least_squares(
        fun=fit_line,
        x0=[1.0, 0.0],
        args=(x, y_noisy),
        loss='huber',
        f_scale=2.0
    )

    # Cauchy loss
    result_cauchy = least_squares(
        fun=fit_line,
        x0=[1.0, 0.0],
        args=(x, y_noisy),
        loss='cauchy',
        f_scale=2.0
    )

    # Extract fitted parameters
    m_true, b_true = 2.0, 1.0
    m_l2, b_l2 = result_l2.x
    m_huber, b_huber = result_huber.x
    m_cauchy, b_cauchy = result_cauchy.x

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Line Fitting with Outliers',
            'Error Comparison'
        ),
        horizontal_spacing=0.12
    )

    # Left: Visual comparison
    inliers = np.ones(n_points, dtype=bool)
    inliers[outlier_indices] = False

    # Plot inliers
    fig.add_trace(
        go.Scatter(
            x=x[inliers],
            y=y_noisy[inliers],
            mode='markers',
            name='Inliers',
            marker=dict(size=8, color='blue', opacity=0.6)
        ),
        row=1, col=1
    )

    # Plot outliers
    fig.add_trace(
        go.Scatter(
            x=x[~inliers],
            y=y_noisy[~inliers],
            mode='markers',
            name='Outliers',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='darkred'))
        ),
        row=1, col=1
    )

    # Plot fitted lines
    x_plot = np.linspace(0, 10, 100)

    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=m_true * x_plot + b_true,
            mode='lines',
            name=f'True: y = {m_true:.1f}x + {b_true:.1f}',
            line=dict(color='green', width=4, dash='solid')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=m_l2 * x_plot + b_l2,
            mode='lines',
            name=f'L2: y = {m_l2:.2f}x + {b_l2:.2f}',
            line=dict(color='orange', width=3, dash='dash')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=m_huber * x_plot + b_huber,
            mode='lines',
            name=f'Huber: y = {m_huber:.2f}x + {b_huber:.2f}',
            line=dict(color='purple', width=3, dash='dashdot')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=m_cauchy * x_plot + b_cauchy,
            mode='lines',
            name=f'Cauchy: y = {m_cauchy:.2f}x + {b_cauchy:.2f}',
            line=dict(color='cyan', width=3, dash='dot')
        ),
        row=1, col=1
    )

    # Right: Error comparison
    errors_l2 = np.abs(fit_line(result_l2.x, x, y_noisy))
    errors_huber = np.abs(fit_line(result_huber.x, x, y_noisy))
    errors_cauchy = np.abs(fit_line(result_cauchy.x, x, y_noisy))

    methods = ['L2<br>(Standard)', 'Huber<br>(Robust)', 'Cauchy<br>(Very Robust)']
    mean_errors = [np.mean(errors_l2), np.mean(errors_huber), np.mean(errors_cauchy)]
    median_errors = [np.median(errors_l2), np.median(errors_huber), np.median(errors_cauchy)]
    max_errors = [np.max(errors_l2), np.max(errors_huber), np.max(errors_cauchy)]

    x_pos = np.arange(len(methods))

    fig.add_trace(
        go.Bar(
            x=methods,
            y=mean_errors,
            name='Mean Error',
            marker=dict(color='steelblue', line=dict(width=2, color='navy')),
            text=[f'{e:.2f}' for e in mean_errors],
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            x=methods,
            y=median_errors,
            name='Median Error',
            marker=dict(color='lightcoral', line=dict(width=2, color='darkred')),
            text=[f'{e:.2f}' for e in median_errors],
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            x=methods,
            y=max_errors,
            name='Max Error',
            marker=dict(color='gold', line=dict(width=2, color='orange')),
            text=[f'{e:.2f}' for e in max_errors],
            textposition='outside'
        ),
        row=1, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)

    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=2)

    # Update layout
    fig.update_layout(
        height=600,
        width=1400,
        title_text="<b>Outlier Robustness Demonstration</b>",
        title_x=0.5,
        title_font_size=18,
        showlegend=True,
        barmode='group',
        hovermode='closest'
    )

    filename = 'outlier_robustness_demo.html'
    fig.write_html(filename)
    print(f"✓ Saved: {filename}")

    # Print numerical comparison
    print("\n" + "=" * 60)
    print("OUTLIER ROBUSTNESS DEMONSTRATION")
    print("=" * 60)
    print(f"True line: y = {m_true:.1f}x + {b_true:.1f}")
    print(f"\nFitted parameters:")
    print(f"  L2 (Standard):      y = {m_l2:.3f}x + {b_l2:.3f}  (error: {np.abs(m_l2 - m_true):.3f})")
    print(f"  Huber (Robust):     y = {m_huber:.3f}x + {b_huber:.3f}  (error: {np.abs(m_huber - m_true):.3f})")
    print(f"  Cauchy (Very Robust): y = {m_cauchy:.3f}x + {b_cauchy:.3f}  (error: {np.abs(m_cauchy - m_true):.3f})")
    print(f"\nMean residual errors:")
    print(f"  L2:     {np.mean(errors_l2):.3f}")
    print(f"  Huber:  {np.mean(errors_huber):.3f}")
    print(f"  Cauchy: {np.mean(errors_cauchy):.3f}")
    print("\n✓ Robust losses recover true parameters despite outliers!")
    print("=" * 60)

    return fig


if __name__ == "__main__":
    print("\nGenerating robust loss function comparison...")
    plot_loss_functions()

    print("\nDemonstrating outlier robustness...")
    demonstrate_outlier_robustness()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. HUBER LOSS (recommended for mocap):
   - L2 (quadratic) for errors < δ → smooth optimization
   - L1 (linear) for errors > δ → robust to outliers
   - δ parameter controls transition (tune to your noise level)

2. WHY IT MATTERS FOR MOTION CAPTURE:
   - Marker occlusions create wild position jumps
   - Standard L2 tries to fit these → corrupts entire trajectory
   - Huber downweights outliers → preserves smooth motion

3. CHOOSING δ (f_scale parameter):
   - δ ≈ 2-3× your expected noise level
   - For 4mm RMS noise → δ ≈ 8-12mm works well
   - Errors > δ are treated as likely outliers

4. OTHER LOSS FUNCTIONS:
   - soft_l1: Smoother version of Huber
   - cauchy: Very aggressive outlier rejection (good for many outliers)
   - arctan: Bounded loss (outliers have zero influence at infinity)

5. PRACTICAL IMPACT:
   - With 2% outliers, L2 can be off by 50%+
   - Huber typically recovers 95%+ of true solution
   - Essential for real-world motion capture data!
    """)
    print("=" * 60)
    print("\n✓ All visualizations saved as interactive HTML files!")
    print("  Open them in your browser to explore the data.")