import logging
import os
from datetime import datetime
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.interpolate import make_interp_spline

# Load environment variables from .env file if available
load_dotenv()

# Global configuration variables (configurable via environment variables)
# Changed from single CSV file to list of files
CSV_FILENAMES: List[str] = os.getenv("CSV_FILENAMES", "0.5C 21700.csv,processed_1C 21700.csv,processed_2C 21700.csv").split(",")
PLOT_DIRECTORY: str = os.getenv("PLOT_DIRECTORY", "plots")
MOVING_AVERAGE_DEFAULT_WINDOW: int = int(os.getenv("MOVING_AVERAGE_DEFAULT_WINDOW", "15"))
RESAMPLE_REDUCTION_FACTOR: float = float(os.getenv("RESAMPLE_REDUCTION_FACTOR", "0.5"))
SMOOTH_CURVE_POINTS: int = int(os.getenv("SMOOTH_CURVE_POINTS", "1000"))
BATTERY_CAPACITY_AH: float = float(os.getenv("BATTERY_CAPACITY_AH", "5"))  # Default battery capacity in Ah

# Colors for different datasets in plots
PLOT_COLORS: List[str] = ["b", "r", "g", "m", "c", "y", "k", "orange"]

# Configure logging for detailed execution tracing
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def time_to_seconds(time_str: str) -> int:
    """Converts a time string in 'HH:MM:SS' format to seconds.

    Args:
        time_str: Time formatted as a string 'HH:MM:SS'.

    Returns:
        Total seconds as an integer. Returns 0 if conversion fails.
    """
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        logging.warning("Failed to convert time string '%s'. Returning 0.", time_str)
        return 0


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Applies a moving average filter over the data.

    Args:
        data: Array of data points.
        window_size: Number of points to include in the moving average window.

    Returns:
        Smoothed data as a NumPy array.
    """
    logging.debug("Applying moving average with window size %d.", window_size)
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def resample_data(x: np.ndarray, y: np.ndarray, factor: float = RESAMPLE_REDUCTION_FACTOR) -> Tuple[np.ndarray, np.ndarray]:
    """Resamples data points reducing the total number based on a given factor.

    Args:
        x: Array of x-coordinates.
        y: Array of y-coordinates.
        factor: Fraction of points to retain (e.g. 0.01 for 1%).

    Returns:
        Tuple containing the resampled x and y arrays.
    """
    n_points: int = max(int(len(x) * factor), 2)  # Ensure at least two points
    indices = np.linspace(0, len(x) - 1, n_points, dtype=int)
    logging.debug("Resampling data from %d to %d points.", len(x), n_points)
    return x[indices], y[indices]


def smooth_curve(x: np.ndarray, y: np.ndarray, n_points: int = SMOOTH_CURVE_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    """Smooths the curve using cubic spline interpolation.

    Args:
        x: Array of x-coordinates.
        y: Array of y-coordinates.
        n_points: Number of interpolation points for smoothing.

    Returns:
        Tuple containing the new x array and its corresponding smoothed y values.
    """
    x_new: np.ndarray = np.linspace(np.min(x), np.max(x), n_points)
    spline = make_interp_spline(x, y, k=3)
    y_new: np.ndarray = spline(x_new)
    logging.debug("Spline interpolation generated %d points.", n_points)
    return x_new, y_new


def apply_moving_average(
    x: np.ndarray, y: np.ndarray, default_window: int = MOVING_AVERAGE_DEFAULT_WINDOW
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies moving average filtering to data and trims the x array accordingly.

    Args:
        x: Original x array.
        y: Original y array.
        default_window: Default window size for moving average filtering.

    Returns:
        Tuple containing the trimmed x array and the moving-averaged y array.
    """
    window_size: int = min(default_window, len(y) // 10) if len(y) > 0 else default_window
    if len(y) > window_size:
        y_ma: np.ndarray = moving_average(y, window_size)
        if window_size % 2 == 1:
            trim: int = (window_size - 1) // 2
            x_ma: np.ndarray = x[trim: -trim]
        else:
            trim_left: int = window_size // 2
            trim_right: int = (window_size // 2) - 1
            x_ma: np.ndarray = x[trim_left: -trim_right]
        logging.debug("Applied moving average; trimmed data to %d points.", len(y_ma))
        return x_ma, y_ma
    logging.debug("Skipping moving average due to insufficient data length.")
    return x, y


def ensure_plot_directory(directory: str = PLOT_DIRECTORY) -> None:
    """Ensures that the directory for saving plots exists.

    Args:
        directory: Directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Created plot directory: %s", directory)
    else:
        logging.debug("Plot directory already exists: %s", directory)


def plot_voltage_vs_capacity(discharge_df: pd.DataFrame, ax=None) -> None:
    """Generates and saves a plot of voltage versus discharge capacity.

    Args:
        discharge_df: DataFrame containing discharge data with 'cap_ah' and 'voltage'.
        ax: Matplotlib axis for plotting. If None, a new figure is created.
    """
    logging.info("Generating Voltage vs Discharge Capacity plot.")

    x_capacity: np.ndarray = discharge_df["cap_ah"].values
    y_voltage: np.ndarray = discharge_df["voltage"].values

    # Smooth the data with moving average and resampling
    x_smoothed, y_smoothed = apply_moving_average(x_capacity, y_voltage)
    x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)

    # Create new figure if no axis provided
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.plot(x_sampled, y_sampled, "b-", linewidth=2)

    # Identify extremum points
    max_index: int = int(np.argmax(y_sampled))
    min_index: int = int(np.argmin(y_sampled))

    ax.plot(x_sampled[max_index], y_sampled[max_index], "ro", markersize=8)
    ax.plot(x_sampled[min_index], y_sampled[min_index], "go", markersize=8)

    ax.annotate(
        f"Min: ({x_sampled[max_index]:.2f}Ah, {y_sampled[max_index]:.2f}V)",
        xy=(x_sampled[max_index], y_sampled[max_index]),
        xytext=(x_sampled[max_index] - 0.1, y_sampled[max_index] + 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )
    ax.annotate(
        f"Max: ({x_sampled[min_index]:.2f}Ah, {y_sampled[min_index]:.2f}V)",
        xy=(x_sampled[min_index], y_sampled[min_index]),
        xytext=(x_sampled[min_index] + 0.1, y_sampled[min_index] - 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )

    ax.set_xlabel("Discharge Capacity (Ah)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.set_title("Voltage vs Discharge Capacity", fontsize=14)
    ax.grid(True)

    plot_path: str = os.path.join(PLOT_DIRECTORY, "voltage_vs_capacity.png")
    plt.savefig(plot_path, dpi=300)
    logging.info("Saved Voltage vs Capacity plot to %s", plot_path)


def plot_capacity_vs_time(discharge_df: pd.DataFrame, ax=None) -> None:
    """Generates and saves a plot of discharge capacity versus time in seconds.

    Args:
        discharge_df: DataFrame containing discharge data with 'time_seconds' and 'cap_ah'.
        ax: Matplotlib axis for plotting. If None, a new figure is created.
    """
    logging.info("Generating Discharge Capacity vs Time plot.")

    x_time: np.ndarray = discharge_df["time_seconds"].values
    y_capacity: np.ndarray = discharge_df["cap_ah"].values

    # Smooth the data with moving average and resampling
    x_smoothed, y_smoothed = apply_moving_average(x_time, y_capacity)
    x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)

    # Create new figure if no axis provided
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.plot(x_sampled, y_sampled, "r-", linewidth=2)

    # Identify extremum points
    max_index: int = int(np.argmax(y_sampled))
    min_index: int = int(np.argmin(y_sampled))

    ax.plot(x_sampled[max_index], y_sampled[max_index], "bo", markersize=8)
    ax.plot(x_sampled[min_index], y_sampled[min_index], "go", markersize=8)

    ax.annotate(
        f"Max: ({x_sampled[max_index]:.0f}s, {y_sampled[max_index]:.2f}Ah)",
        xy=(x_sampled[max_index], y_sampled[max_index]),
        xytext=(x_sampled[max_index] - 100, y_sampled[max_index] + 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )
    ax.annotate(
        f"Min: ({x_sampled[min_index]:.0f}s, {y_sampled[min_index]:.2f}Ah)",
        xy=(x_sampled[min_index], y_sampled[min_index]),
        xytext=(x_sampled[min_index] + 100, y_sampled[min_index] - 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Discharge Capacity (Ah)", fontsize=12)
    ax.set_title("Discharge Capacity vs Time", fontsize=14)
    ax.grid(True)

    plot_path: str = os.path.join(PLOT_DIRECTORY, "capacity_vs_time.png")
    plt.savefig(plot_path, dpi=300)
    logging.info("Saved Capacity vs Time plot to %s", plot_path)


def plot_temperature_vs_time(discharge_df: pd.DataFrame, ax=None) -> None:
    """Generates and saves a plot of temperature versus time in seconds.

    Args:
        discharge_df: DataFrame containing discharge data with 'time_seconds' and 'temperature'.
        ax: Matplotlib axis for plotting. If None, a new figure is created.
    """
    logging.info("Generating Temperature vs Time plot.")

    x_time: np.ndarray = discharge_df["time_seconds"].values
    y_temperature: np.ndarray = discharge_df["temp"].values

    # Smooth the data with moving average and resampling
    x_smoothed, y_smoothed = apply_moving_average(x_time, y_temperature)
    x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)

    # Create new figure if no axis provided
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.plot(x_sampled, y_sampled, "g-", linewidth=2)

    # Identify extremum points
    max_index: int = int(np.argmax(y_sampled))
    min_index: int = int(np.argmin(y_sampled))

    ax.plot(x_sampled[max_index], y_sampled[max_index], "ro", markersize=8)
    ax.plot(x_sampled[min_index], y_sampled[min_index], "bo", markersize=8)

    ax.annotate(
        f"Max: ({x_sampled[max_index]:.0f}s, {y_sampled[max_index]:.2f}°C)",
        xy=(x_sampled[max_index], y_sampled[max_index]),
        xytext=(x_sampled[max_index] - 100, y_sampled[max_index] + 0.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )
    ax.annotate(
        f"Min: ({x_sampled[min_index]:.0f}s, {y_sampled[min_index]:.2f}°C)",
        xy=(x_sampled[min_index], y_sampled[min_index]),
        xytext=(x_sampled[min_index] + 100, y_sampled[min_index] - 0.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title("Temperature vs Time", fontsize=14)
    ax.grid(True)

    plot_path: str = os.path.join(PLOT_DIRECTORY, "temperature_vs_time.png")
    plt.savefig(plot_path, dpi=300)
    logging.info("Saved Temperature vs Time plot to %s", plot_path)


def plot_soc_vs_time(discharge_df: pd.DataFrame, ax=None) -> None:
    """Generates and saves a plot of State of Charge (SoC) versus time in seconds.

    Args:
        discharge_df: DataFrame containing discharge data with 'time_seconds' and 'cap_ah'.
        ax: Matplotlib axis for plotting. If None, a new figure is created.
    """
    logging.info("Generating State of Charge vs Time plot.")

    x_time: np.ndarray = discharge_df["time_seconds"].values
    # Calculate SoC as percentage: (current charge / total capacity) * 100
    y_soc: np.ndarray = (1 - (discharge_df["cap_ah"].values / BATTERY_CAPACITY_AH)) * 100

    # Smooth the data with moving average and resampling
    x_smoothed, y_smoothed = apply_moving_average(x_time, y_soc)
    x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)

    # Create new figure if no axis provided
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.plot(x_sampled, y_sampled, "m-", linewidth=2)

    # Identify extremum points
    max_index: int = int(np.argmax(y_sampled))
    min_index: int = int(np.argmin(y_sampled))

    ax.plot(x_sampled[max_index], y_sampled[max_index], "bo", markersize=8)
    ax.plot(x_sampled[min_index], y_sampled[min_index], "go", markersize=8)

    ax.annotate(
        f"Max: ({x_sampled[max_index]:.0f}s, {y_sampled[max_index]:.1f}%)",
        xy=(x_sampled[max_index], y_sampled[max_index]),
        xytext=(x_sampled[max_index] - 100, y_sampled[max_index] + 2),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )
    ax.annotate(
        f"Min: ({x_sampled[min_index]:.0f}s, {y_sampled[min_index]:.1f}%)",
        xy=(x_sampled[min_index], y_sampled[min_index]),
        xytext=(x_sampled[min_index] + 100, y_sampled[min_index] - 2),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("State of Charge (%)", fontsize=12)
    ax.set_title("Battery State of Charge vs Time", fontsize=14)
    ax.grid(True)

    plot_path: str = os.path.join(PLOT_DIRECTORY, "soc_vs_time.png")
    plt.savefig(plot_path, dpi=300)
    logging.info("Saved State of Charge vs Time plot to %s", plot_path)


def read_csv_data(filenames: List[str]) -> Dict[str, pd.DataFrame]:
    """Reads CSV data from multiple files into separate DataFrames.

    Args:
        filenames: List of paths to the CSV files.

    Returns:
        A dictionary mapping filenames to their respective DataFrames.
    """
    data_dict = {}
    for filename in filenames:
        logging.info("Reading CSV file: %s", filename)
        try:
            df = pd.read_csv(filename)
            # Extract the base filename without path for use as label
            base_name = os.path.basename(filename)
            # Remove "processed_" prefix if it exists
            if base_name.startswith("processed_"):
                base_name = base_name[10:]
            data_dict[base_name] = df
            logging.info("Successfully read data from %s", filename)
        except Exception as e:
            logging.error("Failed to read file %s: %s", filename, str(e))
    
    logging.info("Read data from %d files.", len(data_dict))
    return data_dict


def prepare_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Prepares and filters multiple DataFrames for plotting.

    Converts time strings to seconds and filters each DataFrame for active discharge data.

    Args:
        data_dict: Dictionary mapping filenames to their DataFrames.

    Returns:
        Dictionary with filtered DataFrames containing data where discharge is active.
    """
    filtered_dict = {}
    for label, df in data_dict.items():
        logging.info(f"Converting time to seconds for dataset: {label}")
        df["time_seconds"] = df["time"].apply(time_to_seconds)
        # Filter rows where discharge is active (is_on == 1.0)
        filtered_df = df[df["is_on"] == 1.0].copy()
        logging.info(f"Filtered data for {label} to {len(filtered_df)} active discharge rows.")
        filtered_dict[label] = filtered_df
    
    return filtered_dict


def main() -> None:
    """Main function to process data, generate plots, and save them."""
    logging.info("Program started.")

    # Ensure plot directory exists
    ensure_plot_directory()

    # Read and prepare the data from multiple CSV files
    data_dict = read_csv_data(CSV_FILENAMES)
    filtered_data_dict = prepare_data(data_dict)
    
    logging.info(f"Successfully loaded and prepared {len(filtered_data_dict)} datasets.")

    # Generate individual multi-dataset plots
    plot_voltage_vs_capacity_multi(filtered_data_dict)
    plot_capacity_vs_time_multi(filtered_data_dict)
    plot_temperature_vs_time_multi(filtered_data_dict)
    plot_soc_vs_time_multi(filtered_data_dict)
    
    # Generate combined multi-dataset plot
    plot_combined_multi_datasets(filtered_data_dict)

    logging.info("All multi-dataset plots generated and saved successfully.")
    
    # For backward compatibility, if only one CSV file is specified,
    # also generate the original single-dataset plots
    if len(filtered_data_dict) == 1:
        logging.info("Only one dataset specified, also generating single-dataset plots.")
        discharge_df = next(iter(filtered_data_dict.values()))
        
        # Create a figure with a 2x2 grid layout for all four single-dataset plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        
        # Generate plots using the grid layout
        plot_voltage_vs_capacity(discharge_df, axes[0, 0])    # top-left
        plot_soc_vs_time(discharge_df, axes[0, 1])           # top-right (switched with capacity vs time)
        plot_temperature_vs_time(discharge_df, axes[1, 0])    # bottom-left
        plot_capacity_vs_time(discharge_df, axes[1, 1])       # bottom-right (switched with soc vs time)
        
        # Set a title for the entire figure
        fig.suptitle('Battery Discharge Analysis - Single Dataset', fontsize=16)
        
        # Save the combined figure
        combined_plot_path = os.path.join(PLOT_DIRECTORY, "combined_plots.png")
        plt.savefig(combined_plot_path, dpi=300)
        logging.info("Saved single-dataset combined plots to %s", combined_plot_path)

    # Display all plots in a single window
    plt.show()

    logging.info("Program completed successfully.")


def plot_voltage_vs_capacity_multi(data_dict: Dict[str, pd.DataFrame], save=True) -> plt.Figure:
    """Generates a plot of voltage versus discharge capacity for multiple datasets.

    Args:
        data_dict: Dictionary mapping dataset labels to their DataFrames.
        save: Whether to save the plot to a file.

    Returns:
        The matplotlib Figure object.
    """
    logging.info("Generating multi-dataset Voltage vs Discharge Capacity plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (label, df) in enumerate(data_dict.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        
        x_capacity = df["cap_ah"].values
        y_voltage = df["voltage"].values
        
        # Smooth the data with moving average and resampling
        x_smoothed, y_smoothed = apply_moving_average(x_capacity, y_voltage)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        
        ax.plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
    
    ax.set_xlabel("Discharge Capacity (Ah)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.set_title("Voltage vs Discharge Capacity", fontsize=14)
    ax.grid(True)
    ax.legend(loc='best')
    
    if save:
        plot_path = os.path.join(PLOT_DIRECTORY, "voltage_vs_capacity_multi.png")
        plt.savefig(plot_path, dpi=300)
        logging.info("Saved multi-dataset Voltage vs Capacity plot to %s", plot_path)
    
    return fig


def plot_capacity_vs_time_multi(data_dict: Dict[str, pd.DataFrame], save=True) -> plt.Figure:
    """Generates a plot of discharge capacity versus time for multiple datasets.

    Args:
        data_dict: Dictionary mapping dataset labels to their DataFrames.
        save: Whether to save the plot to a file.

    Returns:
        The matplotlib Figure object.
    """
    logging.info("Generating multi-dataset Discharge Capacity vs Time plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (label, df) in enumerate(data_dict.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        
        x_time = df["time_seconds"].values
        y_capacity = df["cap_ah"].values
        
        # Smooth the data with moving average and resampling
        x_smoothed, y_smoothed = apply_moving_average(x_time, y_capacity)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        
        ax.plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
    
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Discharge Capacity (Ah)", fontsize=12)
    ax.set_title("Discharge Capacity vs Time", fontsize=14)
    ax.grid(True)
    ax.legend(loc='best')
    
    if save:
        plot_path = os.path.join(PLOT_DIRECTORY, "capacity_vs_time_multi.png")
        plt.savefig(plot_path, dpi=300)
        logging.info("Saved multi-dataset Capacity vs Time plot to %s", plot_path)
    
    return fig


def plot_temperature_vs_time_multi(data_dict: Dict[str, pd.DataFrame], save=True) -> plt.Figure:
    """Generates a plot of temperature versus time for multiple datasets.

    Args:
        data_dict: Dictionary mapping dataset labels to their DataFrames.
        save: Whether to save the plot to a file.

    Returns:
        The matplotlib Figure object.
    """
    logging.info("Generating multi-dataset Temperature vs Time plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (label, df) in enumerate(data_dict.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        
        x_time = df["time_seconds"].values
        y_temperature = df["temp"].values
        
        # Smooth the data with moving average and resampling
        x_smoothed, y_smoothed = apply_moving_average(x_time, y_temperature)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        
        ax.plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
    
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title("Temperature vs Time", fontsize=14)
    ax.grid(True)
    ax.legend(loc='best')
    
    if save:
        plot_path = os.path.join(PLOT_DIRECTORY, "temperature_vs_time_multi.png")
        plt.savefig(plot_path, dpi=300)
        logging.info("Saved multi-dataset Temperature vs Time plot to %s", plot_path)
    
    return fig


def plot_soc_vs_time_multi(data_dict: Dict[str, pd.DataFrame], save=True) -> plt.Figure:
    """Generates a plot of State of Charge versus time for multiple datasets.

    Args:
        data_dict: Dictionary mapping dataset labels to their DataFrames.
        save: Whether to save the plot to a file.

    Returns:
        The matplotlib Figure object.
    """
    logging.info("Generating multi-dataset State of Charge vs Time plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (label, df) in enumerate(data_dict.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        
        x_time = df["time_seconds"].values
        # Calculate SoC as percentage: (current charge / total capacity) * 100
        y_soc = (1 - (df["cap_ah"].values / BATTERY_CAPACITY_AH)) * 100
        
        # Smooth the data with moving average and resampling
        x_smoothed, y_smoothed = apply_moving_average(x_time, y_soc)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        
        ax.plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
    
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("State of Charge (%)", fontsize=12)
    ax.set_title("Battery State of Charge vs Time", fontsize=14)
    ax.grid(True)
    ax.legend(loc='best')
    
    if save:
        plot_path = os.path.join(PLOT_DIRECTORY, "soc_vs_time_multi.png")
        plt.savefig(plot_path, dpi=300)
        logging.info("Saved multi-dataset State of Charge vs Time plot to %s", plot_path)
    
    return fig


def plot_combined_multi_datasets(data_dict: Dict[str, pd.DataFrame]) -> plt.Figure:
    """Creates a 2x2 grid of all plots for multiple datasets.
    
    Args:
        data_dict: Dictionary mapping dataset labels to their DataFrames.
        
    Returns:
        The matplotlib Figure object.
    """
    logging.info("Generating combined plot with all datasets.")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    
    # Counter for assigning consistent colors across all plots
    for i, (label, df) in enumerate(data_dict.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        
        # Voltage vs Capacity (top-left)
        x_capacity = df["cap_ah"].values
        y_voltage = df["voltage"].values
        x_smoothed, y_smoothed = apply_moving_average(x_capacity, y_voltage)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        axes[0, 0].plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
        
        # SoC vs Time (top-right) - Switched with Capacity vs Time
        x_time = df["time_seconds"].values
        y_soc = (1 - (df["cap_ah"].values / BATTERY_CAPACITY_AH)) * 100
        x_smoothed, y_smoothed = apply_moving_average(x_time, y_soc)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        axes[0, 1].plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
        
        # Temperature vs Time (bottom-left)
        y_temperature = df["temp"].values
        x_smoothed, y_smoothed = apply_moving_average(x_time, y_temperature)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        axes[1, 0].plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
        
        # Capacity vs Time (bottom-right) - Switched with SoC vs Time
        y_capacity = df["cap_ah"].values
        x_smoothed, y_smoothed = apply_moving_average(x_time, y_capacity)
        x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)
        axes[1, 1].plot(x_sampled, y_sampled, color=color, linewidth=2, label=label)
    
    # Customize each subplot
    axes[0, 0].set_xlabel("Discharge Capacity (Ah)")
    axes[0, 0].set_ylabel("Voltage (V)")
    axes[0, 0].set_title("Voltage vs Discharge Capacity")
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc='best')
    
    # Updated titles and labels after switching positions
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("State of Charge (%)")
    axes[0, 1].set_title("Battery State of Charge vs Time")
    axes[0, 1].grid(True)
    axes[0, 1].legend(loc='best')
    
    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 0].set_ylabel("Temperature (°C)")
    axes[1, 0].set_title("Temperature vs Time")
    axes[1, 0].grid(True)
    axes[1, 0].legend(loc='best')
    
    # Updated titles and labels after switching positions
    axes[1, 1].set_xlabel("Time (seconds)")
    axes[1, 1].set_ylabel("Discharge Capacity (Ah)")
    axes[1, 1].set_title("Discharge Capacity vs Time")
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc='best')
    
    # Set a title for the entire figure
    fig.suptitle('Battery Discharge Analysis - Multiple Datasets', fontsize=16)
    
    # Save the combined figure
    plot_path = os.path.join(PLOT_DIRECTORY, "combined_plots_multi.png")
    plt.savefig(plot_path, dpi=300)
    logging.info("Saved combined multi-dataset plots to %s", plot_path)
    
    return fig


if __name__ == "__main__":
    main()