# SAXS Data Averager and Splitter

A Streamlit-based graphical application for processing, filtering, and averaging Small-Angle X-ray Scattering (SAXS) data files.

## Overview

This project provides two main tools:
1. **SAXS Data Averager (`SAXS_averager.py`)**: The primary application for loading multiple SAXS `.dat` or `.csv` files, automatically identifying and filtering out anomalous data points or bad frames, and calculating a clean average curve.
2. **SAXS File Splitter (`SAXS_splitter.py`)**: A supplementary utility that helps users organize their raw data by parsing filenames and moving files into specific subdirectories before averaging.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install streamlit pandas numpy plotly
```

## Usage

### Running the Averager
Start the main application by running:
```bash
streamlit run SAXS_averager.py
```

**Features:**
- **Directory Navigation:** Browse and select the directory containing your SAXS data.
- **Data Filtering:** Include or exclude files using string matching, or manually select files from a list.
- **Data Chopping:** Remove the first N points (often noisy beamstop data) from every curve.
- **Automatic Masking:**
  - **Mask %:** Points that deviate from the median curve by more than this percentage are flagged as outliers and excluded from the average.
  - **Ignore %:** Entire frames/files where the number of masked points exceeds this percentage are excluded completely.
- **Manual Overrides:** Use the interactive table to explicitly ignore (or force include) specific frames.
- **Visualization:** View curves in Lin-Lin, Log-Log, Guinier, Kratky, or Porod modes. Outlier points are grayed out.
- **Exporting:** Save the resulting averaged data (`q`, `I_mean`, `I_std`) along with a detailed log header.
- **Batch Processing:** Automatically analyze all subdirectories of a chosen root folder and save the resulting averages to a target directory.

### Running the Splitter
You can launch the File Splitter directly from a button inside the Averager's sidebar, or run it standalone:
```bash
streamlit run SAXS_splitter.py
```

**Features:**
- **Auto-Categorization**: Scans a directory and suggests target folders based on text extracted from the filenames.
- **Interactive Reassignment**: Edit the "Target Directory" in the table to change where files will go.
- **Bulk Move**: Safely creates target folders and moves the files.

## State Persistence
The application saves your preferences, directory locations, and manual frame overrides into `saxs_averager_state.json`. If you open the app later, it will load your previous configuration automatically.
