from typing import Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class LineType(Enum):
    """A convenience wrapper around plotlys `dash` prop."""

    DASHED = "dash"
    SOLID = None


class TrajectoryPlotter:
    """Class to generate traces on and format a plot.


    Usage:
    ```python
    fig = go.Figure()
    plotter = TrajectoryPlotter(...) # instantiate the plotter
    # plot some traces
    line = plotter.line_trace(...)
    markers = plotter.marker_trace(...)
    # add the traces to the figure in the desired order
    fig.add_trace(line)
    fig.add_trace(markers)
    # format the figure
    plotter.format_figure(...)
    # show the figure
    fig.show()
    ```
    """

    def __init__(self, dash_width: float = 3.5, fontsize: float = 14) -> None:
        self.dash_width = dash_width
        self.fontsize = fontsize

    def line_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: str,
        line_type: LineType = LineType.SOLID,
        show_legend: bool = True,
        name: str = "",
    ) -> go.Scatter:
        """draws a dashed line plot"""
        return go.Scatter(
            x=x,
            y=y,
            line=dict(color=color, width=self.dash_width, dash=line_type.value),
            showlegend=show_legend,
            name=name,
        )

    def marker_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        symbol: str,
        color: str,
        name: str,
        marker_size: float,
        marker_outline_width: float,
    ) -> go.Scatter:
        """draws a scatter plot using marker symbols"""
        num_markers, *_ = np.shape(x)
        return go.Scatter(
            mode="markers",
            x=x,
            y=y,
            marker_symbol=np.repeat(symbol, num_markers),
            marker_color=color,
            marker_line_color="black",
            marker_line_width=marker_outline_width,
            marker_size=marker_size,
            name=(name),
        )

    def format_figure(
        self,
        fig,
        x_title: str,
        y_title: str,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        width_px: Optional[int] = None,
        height_px: Optional[int] = None,
        figure_title: Optional[str] = None,
    ) -> None:
        # set a title object for the layout, if a figure title was provided, see:
        # https://plotly.com/python/figure-labels/#align-plot-title
        title = dict(
            text=figure_title,
            y=0.9,  # vertical position in image
            x=0.5,  # horizontal position in image
            xanchor="center",
            yanchor="top",
        )

        """styles the plot according to the paper style-guidelines"""
        fig.update_layout(
            # set a title (if provided)
            title=title if figure_title is not None else None,
            # move the legend to the top right
            legend=dict(
                yanchor="top",  # ?
                y=0.99,  # y relative position
                xanchor="right",  # ?
                x=0.99,  # x relative position
                bgcolor="lightgray",  # backround color for the legend
                bordercolor="black",  # border color
                borderwidth=1,  # border width
                font=dict(size=self.fontsize),
            ),
            paper_bgcolor="rgba(0,0,0,0)",  # set background transparent ...
            plot_bgcolor="rgba(0,0,0,0)",  # ... (required for prints)
            # LaTeX axis titles
            yaxis_title=y_title,
            xaxis_title=x_title,
            # plot dimensions
            height=height_px,
            width=width_px,
            font=dict(size=self.fontsize),
            xaxis_range=x_range,
            yaxis_range=y_range,
        )
        # NOTE: zero-lines are styled separately, see:
        # https://plotly.com/python/axes/#styling-and-coloring-axes-and-the-zeroline
        fig.update_xaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            gridcolor="black",
            linecolor="black",
            zerolinecolor="black",  # separate zero line styling
            zerolinewidth=1,
        )
        fig.update_yaxes(
            mirror=True,  # idk why but its needed
            ticks="outside",
            showline=True,
            gridcolor="black",  # grid
            linecolor="black",  # outline
            zerolinecolor="black",
            zerolinewidth=1,
            # these two settings make the axes have the same scale
            scaleanchor="x",
            scaleratio=1,
        )


class MultiHistogramPlotter:
    """Utility class used to plot multiple histograms in aligned plots."""

    def __init__(self, font_size: float = 14) -> None:
        self.font_size = font_size

    def plot_data(self, data: pd.DataFrame, x: str, y: str, color: str) -> None:
        fig = px.histogram(data, x, y, color)
        fig.show()
