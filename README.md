# DL24P Plot Data For Single Cell

A data visualization tool for the ATORCH DL24P battery tester. This project processes CSV files obtained from this [repository](https://github.com/Jay2k1/Electronic_load_DL24.git) containing battery cell discharge data and generates informative plots showing voltage vs. discharge capacity and discharge capacity vs. time relationships.

## Features

- **Data Processing**: Converts time strings to seconds and filters for active discharge data
- **Data Smoothing**: Applies moving average filtering to reduce noise
- **Data Resampling**: Reduces data point density for optimal visualization
- **Curve Smoothing**: Uses cubic spline interpolation for smooth curves
- **Extremum Identification**: Automatically identifies and annotates maximum and minimum points
- **High-Resolution Plots**: Generates publication-quality plots at 300 DPI
- **Environment Variable Configuration**: Customize behavior via environment variables or a .env file
- **Detailed Logging**: Comprehensive logging of all operations for debugging

## Sample Plots

The application generates two main plots:

1. **Voltage vs. Discharge Capacity**: Shows the relationship between battery voltage and discharge capacity
2. **Discharge Capacity vs. Time**: Shows how discharge capacity changes over time

Both plots are saved to the `plots/` directory and displayed on screen.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - pandas
  - matplotlib
  - numpy
  - scipy
  - python-dotenv

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that can significantly speed up the installation process. I recomment that you use uv for this project.

#### Installing uv

1. **Windows (PowerShell)**:

   ```powershell
   curl.exe -L --output uv-installer.ps1 https://astral.sh/uv/install.ps1
   powershell -ex bypass -f uv-installer.ps1
   ```

2. **Windows (Command Prompt)**:

   ```cmd
   curl.exe -L --output uv-installer.ps1 https://astral.sh/uv/install.ps1
   powershell -ex bypass -f uv-installer.ps1
   ```

3. **macOS/Linux**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

#### Installing Requirements with uv

Navigate to the project directory and run:

```bash
uv pip install -r requirements.txt
```

### Using pip (Alternative)

If you prefer to use traditional pip:

```bash
pip install -r requirements.txt
```

## Usage

### Running with uv

```bash
uv run main.py
```

### Running with Python

```bash
python main.py
```

### Using a Different CSV File

You can specify a different CSV file either:

1. **Via environment variable**:

   ```bash
   # Windows
   set CSV_FILENAME=Cell x_raw_20250410_163303.csv
   uv run main.py

   # macOS/Linux
   export CSV_FILENAME=Cell x_raw_20250410_163303.csv
   uv run main.py
   ```

2. **Via .env file**:
   Create a `.env` file in the project root with:
   ```
   CSV_FILENAME=Cell x_raw_20250410_163303.csv
   ```

## Configuration

The following environment variables can be used to customize behavior:

| Variable                      | Default                        | Description                                                      |
| ----------------------------- | ------------------------------ | ---------------------------------------------------------------- |
| CSV_FILENAME                  | Cell x_raw_20250410_175144.csv | Path to the CSV file to process                                  |
| PLOT_DIRECTORY                | plots                          | Directory where plots will be saved                              |
| MOVING_AVERAGE_DEFAULT_WINDOW | 15                             | Window size for moving average filter                            |
| RESAMPLE_REDUCTION_FACTOR     | 0.01                           | Fraction of points to retain when resampling (e.g., 0.01 for 1%) |
| SMOOTH_CURVE_POINTS           | 1000                           | Number of points used for curve smoothing                        |

## CSV File Format

The application expects CSV files with the following columns:

- `time`: Time in HH:MM:SS format
- `voltage`: Battery voltage
- `cap_ah`: Discharge capacity in Amp-hours
- `is_on`: Binary flag (1.0 when discharge is active, 0.0 otherwise)

## Project Structure

```
├── main.py                          # Main application code
├── requirements.txt                 # Project dependencies
├── Cell x_raw_20250410_175144.csv   # Example input data
├── Cell x_raw_20250410_163303.csv   # Alternative input data
└── plots/                           # Generated plot outputs
    ├── capacity_vs_time.png         # Plot of capacity vs. time
    └── voltage_vs_capacity.png      # Plot of voltage vs. capacity
```

## Customizing Plots

The plots can be customized by modifying the `plot_voltage_vs_capacity` and `plot_capacity_vs_time` functions in `main.py`. You can adjust:

- Figure size
- Line colors and widths
- Marker styles
- Annotation positions
- Font sizes
- Plot titles and labels

## Advanced Usage

### Creating a Virtual Environment with uv

```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### Running with Different Parameters

```bash
uv run -m python main.py
```

## Troubleshooting

- **Plot directory not created**: Check if you have write permissions in the current directory
- **CSV file not found**: Verify the file path is correct and accessible
- **Time conversion warnings**: Ensure your CSV file uses the correct HH:MM:SS time format
- **Low resolution plots**: Increase the `SMOOTH_CURVE_POINTS` value in your .env file
