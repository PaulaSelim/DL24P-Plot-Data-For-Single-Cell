import logging
import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.interpolate import make_interp_spline

# Load environment variables from .env file if available
load_dotenv()

# Global configuration variables (configurable via environment variables)
CSV_FILENAME: str = os.getenv("CSV_FILENAME", "Cell x_raw_20250410_175144.csv")
PLOT_DIRECTORY: str = os.getenv("PLOT_DIRECTORY", "plots")
MOVING_AVERAGE_DEFAULT_WINDOW: int = int(os.getenv("MOVING_AVERAGE_DEFAULT_WINDOW", "15"))
RESAMPLE_REDUCTION_FACTOR: float = float(os.getenv("RESAMPLE_REDUCTION_FACTOR", "0.01"))
SMOOTH_CURVE_POINTS: int = int(os.getenv("SMOOTH_CURVE_POINTS", "1000"))

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


def plot_voltage_vs_capacity(discharge_df: pd.DataFrame) -> None:
    """Generates and saves a plot of voltage versus discharge capacity.

    Args:
        discharge_df: DataFrame containing discharge data with 'cap_ah' and 'voltage'.
    """
    logging.info("Generating Voltage vs Discharge Capacity plot.")

    x_capacity: np.ndarray = discharge_df["cap_ah"].values
    y_voltage: np.ndarray = discharge_df["voltage"].values

    # Smooth the data with moving average and resampling
    x_smoothed, y_smoothed = apply_moving_average(x_capacity, y_voltage)
    x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)

    plt.figure(figsize=(10, 6))
    plt.plot(x_sampled, y_sampled, "b-", linewidth=2)

    # Identify extremum points
    max_index: int = int(np.argmax(y_sampled))
    min_index: int = int(np.argmin(y_sampled))

    plt.plot(x_sampled[max_index], y_sampled[max_index], "ro", markersize=8)
    plt.plot(x_sampled[min_index], y_sampled[min_index], "go", markersize=8)

    plt.annotate(
        f"Min: ({x_sampled[max_index]:.2f}Ah, {y_sampled[max_index]:.2f}V)",
        xy=(x_sampled[max_index], y_sampled[max_index]),
        xytext=(x_sampled[max_index] - 0.1, y_sampled[max_index] + 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )
    plt.annotate(
        f"Max: ({x_sampled[min_index]:.2f}Ah, {y_sampled[min_index]:.2f}V)",
        xy=(x_sampled[min_index], y_sampled[min_index]),
        xytext=(x_sampled[min_index] + 0.1, y_sampled[min_index] - 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )

    plt.xlabel("Discharge Capacity (Ah)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title("Voltage vs Discharge Capacity", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    plot_path: str = os.path.join(PLOT_DIRECTORY, "voltage_vs_capacity.png")
    plt.savefig(plot_path, dpi=300)
    logging.info("Saved Voltage vs Capacity plot to %s", plot_path)


def plot_capacity_vs_time(discharge_df: pd.DataFrame) -> None:
    """Generates and saves a plot of discharge capacity versus time in seconds.

    Args:
        discharge_df: DataFrame containing discharge data with 'time_seconds' and 'cap_ah'.
    """
    logging.info("Generating Discharge Capacity vs Time plot.")

    x_time: np.ndarray = discharge_df["time_seconds"].values
    y_capacity: np.ndarray = discharge_df["cap_ah"].values

    # Smooth the data with moving average and resampling
    x_smoothed, y_smoothed = apply_moving_average(x_time, y_capacity)
    x_sampled, y_sampled = resample_data(x_smoothed, y_smoothed)

    plt.figure(figsize=(10, 6))
    plt.plot(x_sampled, y_sampled, "r-", linewidth=2)

    # Identify extremum points
    max_index: int = int(np.argmax(y_sampled))
    min_index: int = int(np.argmin(y_sampled))

    plt.plot(x_sampled[max_index], y_sampled[max_index], "bo", markersize=8)
    plt.plot(x_sampled[min_index], y_sampled[min_index], "go", markersize=8)

    plt.annotate(
        f"Max: ({x_sampled[max_index]:.0f}s, {y_sampled[max_index]:.2f}Ah)",
        xy=(x_sampled[max_index], y_sampled[max_index]),
        xytext=(x_sampled[max_index] - 100, y_sampled[max_index] + 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )
    plt.annotate(
        f"Min: ({x_sampled[min_index]:.0f}s, {y_sampled[min_index]:.2f}Ah)",
        xy=(x_sampled[min_index], y_sampled[min_index]),
        xytext=(x_sampled[min_index] + 100, y_sampled[min_index] - 0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
    )

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Discharge Capacity (Ah)", fontsize=12)
    plt.title("Discharge Capacity vs Time", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    plot_path: str = os.path.join(PLOT_DIRECTORY, "capacity_vs_time.png")
    plt.savefig(plot_path, dpi=300)
    logging.info("Saved Capacity vs Time plot to %s", plot_path)


def read_csv_data(filename: str) -> pd.DataFrame:
    """Reads CSV data into a DataFrame.

    Args:
        filename: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the CSV data.
    """
    logging.info("Reading CSV file: %s", filename)
    df: pd.DataFrame = pd.read_csv(filename)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares and filters the DataFrame for plotting.

    Converts time strings to seconds and filters the DataFrame for active discharge data.

    Args:
        df: The original DataFrame read from the CSV file.

    Returns:
        Filtered DataFrame containing data where discharge is active.
    """
    logging.info("Converting time to seconds.")
    df["time_seconds"] = df["time"].apply(time_to_seconds)
    # Filter rows where discharge is active (is_on == 1.0)
    filtered_df: pd.DataFrame = df[df["is_on"] == 1.0].copy()
    logging.info("Filtered data to %d active discharge rows.", len(filtered_df))
    return filtered_df


def main() -> None:
    """Main function to process data, generate plots, and save them."""
    logging.info("Program started.")

    # Ensure plot directory exists
    ensure_plot_directory()

    # Read and prepare the data
    df: pd.DataFrame = read_csv_data(CSV_FILENAME)
    discharge_df: pd.DataFrame = prepare_data(df)

    # Generate plots
    plot_voltage_vs_capacity(discharge_df)
    plot_capacity_vs_time(discharge_df)

    # Display plots to the user
    plt.show()

    logging.info("All plots generated and saved successfully.")


if __name__ == "__main__":
    main()