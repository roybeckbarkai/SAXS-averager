## SAXS averager 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob
import json
from datetime import datetime
import sys
import subprocess


# --- Page Configuration ---
st.set_page_config(page_title="SAXS Data Averager", layout="wide")

# --- Constants & State Persistence ---
CONFIG_FILE = "saxs_averager_state.json"

def load_state():
    """Load previous session state from disk."""
    defaults = {
        "working_dir": os.getcwd(),
        "mask_percent": 20.0,
        "ignore_percent": 10.0,
        "plot_mode": "Log-Log",
        "file_overrides": {},  # Stores manual Ignore/Masked decisions
        "filter_mode": "All Files",
        "filter_include": "",
        "filter_exclude": "",
        "save_fname": "averaged_saxs_ave.dat"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception:
            pass
    return defaults

def save_state(state_dict):
    """Save current session state to disk."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(state_dict, f)
    except Exception:
        pass

def select_folder():
    """Opens a folder selection dialog and returns the selected path."""
    # 1. Try native macOS dialog via AppleScript (bypasses Python/Tk issues)
    if sys.platform == 'darwin':
        try:
            script = 'POSIX path of (choose folder with prompt "Select Data Directory")'
            cmd = ['osascript', '-e', script]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return "" # User cancelled
        except Exception:
            pass # Fall back to Tkinter if AppleScript fails

    try:
        # These imports are here to avoid issues on systems without tkinter
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window.
        root.attributes('-topmost', True)  # Bring the dialog to the front.
        folder_path = filedialog.askdirectory(parent=root)
        root.destroy()
        return folder_path
    except (ImportError, RuntimeError) as e:
        st.warning(f"Could not open directory browser: {e}. Please paste the path manually.")
        return ""

# Initialize Session State
if "config_loaded" not in st.session_state:
    saved_state = load_state()
    for k, v in saved_state.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state["config_loaded"] = True

# Initialize view_ranges for zoom persistence
if "view_ranges" not in st.session_state:
    st.session_state.view_ranges = {}

# --- Data Loading Logic ---
@st.cache_data(show_spinner=False)
def load_data(directory):
    
    """Parses all .dat/.csv files in the directory."""
    if not os.path.isdir(directory):
        return [], {}, [] # Return q_common, data_map, errors
    
    # Find files
    extensions = ["*.dat", "*.csv", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    files = sorted(list(set(files)))
    
    if not files:
        return [], {}, []
    
    data_map = {}
    q_common = []
    load_errors = [] # Collect errors here
    
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            # Flexible reading: uses specific separators to avoid python engine sniffer issues with trailing spaces
            # Using header=None prevents pandas from accidentally discarding the first data row if no text header exists.
            try:
                sep_regex = ',' if fpath.lower().endswith('.csv') else r'\s+'
                df = pd.read_csv(fpath, sep=sep_regex, engine='python', comment='#', header=None, skip_blank_lines=True)
            except Exception:
                # Fallback to auto-detect if regex separator fails
                df = pd.read_csv(fpath, sep=None, engine='python', comment='#', header=None, skip_blank_lines=True)

            # Check for empty dataframe after parsing comments
            if df.empty:
                load_errors.append(f"'{fname}': File is empty or contains only comments.")
                continue

            # Force numeric and drop NaNs
            df_numeric = df.apply(pd.to_numeric, errors='coerce')

            # Drop rows where q or I are NaN after conversion
            if df_numeric.shape[1] >= 2:
                df_numeric = df_numeric.dropna(subset=[0, 1]) # Ensure q and I columns are numeric

            if df_numeric.empty:
                load_errors.append(f"'{fname}': No valid numeric data rows found after parsing.")
                continue

            # Ensure we have at least Q and I
            if df_numeric.shape[1] < 2:
                load_errors.append(f"'{fname}': Expected at least 2 columns (q, I), but found {df_numeric.shape[1]}.")
                continue
            elif df_numeric.shape[1] > 3:
                load_errors.append(f"'{fname}': Found more than 3 columns ({df_numeric.shape[1]}). Only first two (q, I) will be used.")

            q = df_numeric.iloc[:, 0].values
            i = df_numeric.iloc[:, 1].values

            # Use first file's Q as reference
            if len(q_common) == 0:
                q_common = q
            else:
                # Simple check for matching Q-vector length
                if len(q) != len(q_common):
                    load_errors.append(f"'{fname}': Q-vector length ({len(q)}) does not match reference ({len(q_common)}). Skipping.")
                    continue

            data_map[fname] = {"I": i}
        except pd.errors.EmptyDataError:
            load_errors.append(f"'{fname}': File is empty.")
        except pd.errors.ParserError as e:
            load_errors.append(f"'{fname}': Parsing error - check delimiters or data format. Details: {e}")
        except Exception as e: # Catch any other unexpected errors
            load_errors.append(f"'{fname}': An unexpected error occurred during parsing. Details: {e}")
            continue
            
    return q_common, data_map, load_errors

def calculate_statistics(q, data_map, mask_thresh, ignore_thresh, overrides):
    """
    Core logic:
    1. Calculate Median across valid frames.
    2. Identify bad points (deviation > mask_thresh).
    3. Identify bad frames (bad points > ignore_thresh).
    4. Apply Manual Overrides (Ignore/Masked).
    5. Calculate Final Mean/Std.
    """
    if not data_map:
        return None
    
    filenames = sorted(list(data_map.keys()))
    n_frames = len(filenames)
    n_q = len(q)
    
    # Create matrix: (N_frames x N_q)
    I_matrix = np.zeros((n_frames, n_q))
    for idx, fname in enumerate(filenames):
        I_matrix[idx, :] = data_map[fname]["I"]
        
    # 1. Reference Median 
    # (Exclude manually ignored frames from the median calculation to avoid bias)
    valid_indices = [i for i, f in enumerate(filenames) if not overrides.get(f, {}).get("Ignore", False)]
    if not valid_indices:
        ref_median = np.nanmedian(I_matrix, axis=0)
    else:
        ref_median = np.nanmedian(I_matrix[valid_indices], axis=0)
        
    # 2. Detect Outlier Points
    # Metric: |I - Median| / Median
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs(I_matrix - ref_median)
        rel_diff = diff / (np.abs(ref_median) + 1e-12)
        
    point_mask = rel_diff > (mask_thresh / 100.0)
    
    # 3. Evaluate Frames
    frame_results = []
    final_arrays = []
    
    for idx, fname in enumerate(filenames):
        n_masked = np.sum(point_mask[idx])
        pct_masked = (n_masked / n_q) * 100.0 if n_q > 0 else 0.0
        
        # Auto-detect if frame is "Bad"
        is_auto_excluded = pct_masked > ignore_thresh
        
        # --- Override Logic ---
        user_state = overrides.get(fname, {})
        user_ignore = user_state.get("Ignore", False)
        
        # If user explicitly set "Excluded", use that. Otherwise use Auto result.
        if "Excluded" in user_state:
            is_effective_excluded = user_state["Excluded"]
        else:
            is_effective_excluded = is_auto_excluded
            
        # Final decision: Exclude if Ignored OR (Effective Excluded)
        is_excluded = user_ignore or is_effective_excluded
        
        frame_results.append({
            "Filename": fname,
            "Ignore": user_ignore,
            "Excluded": is_effective_excluded,      # The effective state (User or Auto)
            "Points_Masked": pct_masked > 0,
            "Auto_Exclude_Flag": is_auto_excluded,   # The purely algorithmic state
            "Bad_Points_Pct": pct_masked,
            "Is_Excluded": is_excluded,
            "I": I_matrix[idx],
            "Point_Mask": point_mask[idx]
        })
        
        if not is_excluded:
            # Copy data and NaN out specific masked points for the average
            clean_I = I_matrix[idx].copy()
            clean_I[point_mask[idx]] = np.nan
            final_arrays.append(clean_I)
            
    # 4. Final Aggregation
    if final_arrays:
        stack = np.vstack(final_arrays)
        avg_I = np.nanmean(stack, axis=0)
        std_I = np.nanstd(stack, axis=0)
        final_median = np.nanmedian(stack, axis=0)
    else:
        avg_I = np.zeros(n_q)
        std_I = np.zeros(n_q)
        final_median = np.zeros(n_q)
        
    return {
        "q": q,
        "mean": avg_I,
        "std": std_I,
        "median": final_median,
        "frames": frame_results
    }

# --- GUI Layout ---

st.title("SAXS Data Averager")

# Sidebar
with st.sidebar:
    st.header("Data Source")

    if st.button("Browse for directory"):
        selected_dir = select_folder()
        if selected_dir:
            st.session_state.working_dir = selected_dir
            st.rerun()

    # The text input is bound to session state for robust state management.
    st.text_input("Directory Path", key="working_dir")
    working_dir = st.session_state.working_dir
    

# --- Load Data & Filtering ---
if not os.path.isdir(working_dir):
    st.warning(f"Directory not found: {working_dir}")
    st.stop()

# Modified call to load_data to get errors
q, data_map, file_load_errors = load_data(working_dir)

# Display file loading errors
if file_load_errors:
    st.subheader("File Loading Issues")
    for error_msg in file_load_errors:
        st.error(error_msg)
    st.markdown("---") # Separator for clarity

if not data_map:
    st.warning("No valid .dat, .csv, or .txt files found in directory after filtering and parsing.")
    st.stop()

with st.sidebar:
    st.header("File Filtering")
    # Load filter options from state
    f_mode_default = st.session_state.get("filter_mode", "All Files")
    filter_mode = st.radio("Filter Mode", ["All Files", "Include Pattern", "Exclude Pattern", "Manual Selection"], 
                           index=["All Files", "Include Pattern", "Exclude Pattern", "Manual Selection"].index(f_mode_default) if f_mode_default in ["All Files", "Include Pattern", "Exclude Pattern", "Manual Selection"] else 0)
    
    all_files = sorted(list(data_map.keys()))
    selected_files = all_files

    if filter_mode == "Include Pattern":
        pat_def = st.session_state.get("filter_include", "")
        pat = st.text_input("Filename contains:", pat_def)
        if pat: selected_files = [f for f in all_files if pat in f]
    elif filter_mode == "Exclude Pattern":
        pat_def = st.session_state.get("filter_exclude", "")
        pat = st.text_input("Filename excludes:", pat_def)
        if pat: selected_files = [f for f in all_files if pat not in f]
    elif filter_mode == "Manual Selection":
        selected_files = st.multiselect("Select Files", all_files, default=all_files)
    
    st.caption(f"Selected {len(selected_files)} / {len(all_files)} files")
    data_map = {k: v for k, v in data_map.items() if k in selected_files}
    
    if not data_map:
        st.error("No files selected!")
        st.stop()

    st.header("Data Processing")
    chop_points = st.number_input("Chop first N points", min_value=0, value=0, step=1, help="Remove the first N data points from each file before processing.")

    st.header("Thresholds")
    mask_percent = st.slider("Mask % (deviation from median)", 0.0, 100.0, 
                             st.session_state.get("mask_percent", 20.0), help="Points deviating more than this % from the median are masked.")
    mask_percent = st.number_input("Mask % Value", 0.0, 100.0, value=mask_percent)

    ignore_percent = st.slider("Ignore % (masked points per frame)", 0.0, 100.0, 
                               st.session_state.get("ignore_percent", 10.0), help="Frames with more than this % of masked points are ignored completely.")
    ignore_percent = st.number_input("Ignore % Value", 0.0, 100.0, value=ignore_percent)

    st.header("Visualization")
    plot_modes = ["Lin-Lin", "Log-Log", "Guinier", "Kratky", "Porod"]
    curr_mode = st.session_state.get("plot_mode", "Log-Log")
    if curr_mode not in plot_modes: curr_mode = "Log-Log"
    plot_mode = st.radio("Plot Type", plot_modes, index=plot_modes.index(curr_mode))

    st.info("Tip: Use the table to manually exclude frames or override the masking decision.")

# --- Chop Data Points (Request 2) ---
if chop_points > 0 and q is not None and len(q) > chop_points:
    q = q[chop_points:]
    new_data_map = {}
    for fname, data in data_map.items():
        # Ensure the array is long enough before slicing
        if "I" in data and len(data["I"]) > chop_points:
            new_data_map[fname] = {"I": data["I"][chop_points:]}
    data_map = new_data_map

# 1. Get current overrides (from session state)
current_overrides = st.session_state.get("file_overrides", {})

# 2. Run Calculation
stats = calculate_statistics(q, data_map, mask_percent, ignore_percent, current_overrides)

# 3. Prepare Data for Editor
# We display the *effective* status. User can toggle checkboxes to change it.
frame_data = stats["frames"]

# Assign colors for the plot and table
import plotly.express as px
file_colors = {}
for f in frame_data:
    if f["Is_Excluded"]:
        file_colors[f["Filename"]] = "lightgrey" # Light gray
    else:
        file_colors[f["Filename"]] = "blue"

df_display = pd.DataFrame([
    {
        "Filename": f["Filename"],
        "Ignore": f["Ignore"],
        "Excluded": f["Excluded"],
        "Points Masked": f["Points_Masked"],
        "Bad Points %": f"{f['Bad_Points_Pct']:.2f}%"
    }
    for f in frame_data
])

def color_rows(row):
    color = file_colors.get(row["Filename"], "transparent")
    if color == "lightgrey":
        # Darken text slightly for readability or use background
        # We will set a light background
        return ['background-color: #f0f0f0; color: #a0a0a0'] * len(row)
    return [f'background-color: {color}; color: white; border-radius: 4px;'] * len(row)

styled_df = df_display.style.apply(color_rows, axis=1)


col_ui, col_plot = st.columns([1, 2])

with col_ui:
    st.subheader("Frame Selection")
    st.caption("Check 'Ignore' to force exclude. Check/Uncheck 'Excluded' to override auto-exclusion. 'Points Masked' indicates if any points were skipped.")
    
    edited_df = st.data_editor(
        styled_df,
        column_config={
            "Filename": st.column_config.TextColumn(disabled=True),
            "Bad Points %": st.column_config.TextColumn(disabled=True),
            "Points Masked": st.column_config.CheckboxColumn(disabled=True),
            "Ignore": st.column_config.CheckboxColumn(label="Ignore"),
            "Excluded": st.column_config.CheckboxColumn(label="Excluded")
        },
        disabled=["Filename", "Bad Points %", "Points Masked"],
        hide_index=True,
        key="frame_editor",
        height=600
    )
    
    # Logic to handle edits
    # If the user changes a checkbox, we need to update 'file_overrides' and re-run.
    new_overrides = {}
    has_changes = False
    
    for index, row in edited_df.iterrows():
        fname = row["Filename"]
        u_ign = row["Ignore"]
        u_exc = row["Excluded"]
        
        # Find the original auto-mask value to see if we are overriding
        orig_stat = next(x for x in frame_data if x["Filename"] == fname)
        
        entry = {}
        # Always save Ignore state if True
        if u_ign:
            entry["Ignore"] = True
        
        # Only save Excluded state if it differs from the Algorithm's Auto-Flag
        # This allows the checkbox to update automatically if the user changes the slider,
        # UNLESS the user has manually touched it.
        if u_exc != orig_stat["Auto_Exclude_Flag"]:
            entry["Excluded"] = u_exc
            
        if entry:
            new_overrides[fname] = entry
            
    # Check if overrides differ from session state (triggering a rerun)
    # We do a simplified check. In a real complex app, we might check content equality.
    if new_overrides != current_overrides:
        st.session_state["file_overrides"] = new_overrides
        st.rerun()

with col_plot:
    c1, c2 = st.columns([3, 1])
    c1.subheader("Intensity Plot")
    c2.button("Auto-Zoom to Average", help="Click to reset the view to focus on the average curve.")
    
    fig = go.Figure()
    
    # Helper for transforms
    def transform(q_in, i_in, mode):
        # Avoid log(0)
        i_safe = np.where(i_in > 0, i_in, 1e-12)
        if mode == "Guinier":
            # ln(I) vs q^2
            return q_in**2, np.log(i_safe)
        elif mode == "Kratky":
            # q^2 * I vs q
            return q_in, i_in * (q_in**2)
        elif mode == "Porod":
            # q^4 * I vs q
            return q_in, i_in * (q_in**4)
        else:
            return q_in, i_in

    # Plot Frames
    # Optimization: If > 100 frames, maybe don't plot every single one? 
    # For now, we plot all but make them thin.
    
    for f in stats["frames"]:
        color = file_colors.get(f["Filename"], "rgba(0,0,0,1)")
        
        # We will split data into valid and masked points so masked are dim
        x_p, y_p = transform(stats["q"], f["I"], plot_mode)
        
        # Valid points trace
        point_mask_bool = np.asarray(f["Point_Mask"], dtype=bool)
        valid_mask = ~point_mask_bool
        masked_mask = point_mask_bool
        
        x_valid = np.where(valid_mask, x_p, np.nan)
        y_valid = np.where(valid_mask, y_p, np.nan)
        
        fig.add_trace(go.Scatter(
            x=x_valid, y=y_valid, 
            mode='lines', 
            line=dict(color=color, width=2 if not f["Is_Excluded"] else 1),
            name=f["Filename"],
            showlegend=False, 
            hovertext=f["Filename"],
            hoverinfo='text+x+y'
        ))
        
        # Masked points trace (dim gray)
        if np.any(masked_mask):
            x_masked = np.where(masked_mask, x_p, np.nan)
            y_masked = np.where(masked_mask, y_p, np.nan)
            fname_str = str(f["Filename"])
            fig.add_trace(go.Scatter(
                x=x_masked, y=y_masked, 
                mode='lines', 
                line=dict(color="lightgrey", width=1),
                name=fname_str + " (Masked)",
                showlegend=False, 
                hovertext=fname_str + " (Masked)",
                hoverinfo='text+x+y'
            ))
        
    # Plot Average
    x_avg, y_avg = transform(stats["q"], stats["mean"], plot_mode)
    
    fig.add_trace(go.Scatter(
        x=x_avg, y=y_avg,
        mode='lines',
        line=dict(color='black', width=3),
        name='Average'
    ))
    
    # Axis types
    if plot_mode == "Log-Log":
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    elif plot_mode == "Lin-Lin":
        fig.update_xaxes(type="linear")
        fig.update_yaxes(type="linear")
    
    # Labels
    x_lab = "q [Å⁻¹]"
    y_lab = "I(q)"
    if plot_mode == "Guinier": x_lab, y_lab = "q²", "ln(I)"
    if plot_mode == "Kratky": y_lab = "q² I"
    if plot_mode == "Porod": y_lab = "q⁴ I"
    
    fig.update_layout(
        xaxis_title=x_lab, 
        yaxis_title=y_lab, 
        height=600,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # --- Auto-Zoom Logic (Requests 4, 5, 6) ---
    # Always zoom to the average curve by default. The button above lets the user
    # return to this view after panning/zooming, as any rerun resets the view.
    try:
        # Calculate range from the average trace
        x_min, x_max = np.nanmin(x_avg), np.nanmax(x_avg)
        y_min, y_max = np.nanmin(y_avg), np.nanmax(y_avg)
        
        # Add padding for linear scales
        x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.1
        y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
        
        if plot_mode == "Log-Log":
            # Plotly log axes require the range to be given in log10(value)
            # We must be careful to avoid taking log10 of <= 0
            safe_x_min = np.nanmin(np.where(x_avg > 0, x_avg, np.inf))
            safe_y_min = np.nanmin(np.where(y_avg > 0, y_avg, np.inf))
            x_range = [np.log10(safe_x_min), np.log10(x_max)] if safe_x_min < np.inf and x_max > 0 else None
            y_range = [np.log10(safe_y_min), np.log10(y_max)] if safe_y_min < np.inf and y_max > 0 else None
        else:
            x_range = [x_min - x_pad, x_max + x_pad]
            y_range = [y_min - y_pad, y_max + y_pad]
        
        # For Guinier, ensure x-axis starts at or before 0
        if plot_mode == "Guinier" and x_range is not None:
            x_range[0] = min(x_range[0], 0)

        if x_range is not None:
            fig.update_xaxes(range=x_range)
        if y_range is not None:
            fig.update_yaxes(range=y_range)
    except ValueError: # This happens if x_avg/y_avg is all NaN
        pass # Let plotly auto-range if calculation fails
    st.plotly_chart(fig, width='stretch')
    
    # Export Section
    st.divider()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        dir_name = os.path.basename(os.path.normpath(working_dir))
        if not dir_name:
            dir_name = "averaged_saxs"
        auto_save_name = f"{dir_name}_ave.dat"
        
        save_fname = st.text_input("Output Filename", value=auto_save_name)
        
        # Force ending to be _ave.dat explicitly if user accidentally clears it
        if not save_fname.endswith("_ave.dat"):
            save_fname = save_fname + "_ave.dat"
            st.caption(f"Will be saved as: **{save_fname}**")
    with c2:
        st.write("") # Spacer
        save_btn = st.button("Save Data & Log", type="primary")
    
    if save_btn:
        out_path = os.path.join(working_dir, save_fname)
        
        # Overwrite protection
        proceed = True
        if os.path.exists(out_path):
            if "confirm_overwrite" not in st.session_state:
                st.session_state["confirm_overwrite"] = True
                st.error(f"File exists! Click 'Save' again to confirm overwrite.")
                proceed = False
            else:
                # Reset flag and proceed
                del st.session_state["confirm_overwrite"]
                proceed = True
        
        if proceed:
            # Generate Output
            out_df = pd.DataFrame({
                "q": stats["q"],
                "I_mean": stats["mean"],
                "I_std": stats["std"]
            })
            
            # Generate Log Header
            header_lines = [
                f"SAXS Averaging Log - {datetime.now()}",
                f"Directory: {working_dir}",
                f"Parameters: Mask > {mask_percent}% deviation, Ignore > {ignore_percent}% bad points",
                "-" * 40,
                "Frame Status:"
            ]
            
            for f in stats["frames"]:
                status = "EXCLUDED" if f["Is_Excluded"] else "Included"
                details = []
                if f["Ignore"]: details.append("Manual Ignore")
                if f["Excluded"]: details.append("Auto/Manual Excluded")
                if f["Points_Masked"]: details.append("Some Pts Masked")
                details.append(f"Bad Points: {f['Bad_Points_Pct']:.2f}%")
                header_lines.append(f"{f['Filename']}\t{status}\t[{', '.join(details)}]")
                
            try:
                with open(out_path, "w") as f:
                    for line in header_lines:
                        f.write(f"# {line}\n")
                    out_df.to_csv(f, index=False, sep="\t")
                st.success(f"Successfully saved to {out_path}")
            except Exception as e:
                st.error(f"Error saving file: {e}")

# Save State for Next Run
curr_state = {
    "working_dir": working_dir,
    "mask_percent": mask_percent,
    "ignore_percent": ignore_percent,
    "plot_mode": plot_mode,
    "file_overrides": st.session_state.get("file_overrides", {}),
    "filter_mode": filter_mode,
    "filter_include": pat if filter_mode == "Include Pattern" else st.session_state.get("filter_include", ""),
    "filter_exclude": pat if filter_mode == "Exclude Pattern" else st.session_state.get("filter_exclude", ""),
    "save_fname": save_fname if 'save_fname' in locals() else st.session_state.get("save_fname", "averaged_saxs_ave.dat")
}
save_state(curr_state)
