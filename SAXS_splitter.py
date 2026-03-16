"""
SAXS File Splitter

A Streamlit application that scans a directory of SAXS data files (.dat, .csv),
suggests subdirectories based on patterns in the filenames, and provides an
interactive table to safely move those files into the target directories in bulk.

This is meant to help organize raw data before processing it in SAXS_averager.py.
"""
import streamlit as st
import os
import glob
import re
import shutil
import sys
import subprocess
import pandas as pd
import json

# --- Page Configuration ---
st.set_page_config(page_title="SAXS File Splitter", layout="wide")

# Navigation helper (runs on the client to close the window from the browser side)
def close_window():
    st.markdown("<script>window.close();</script>", unsafe_allow_html=True)
    st.info("Please close this tab to return to the Averager.")

def select_folder():
    """Opens a folder selection dialog and returns the selected path."""
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
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(parent=root)
        root.destroy()
        return folder_path
    except (ImportError, RuntimeError) as e:
        st.warning(f"Could not open directory browser: {e}. Please paste the path manually.")
        return ""

def parse_filename(filename):
    """
    Extracts the 'sample name' from a filename.
    Assumes format starts with numbers, followed by an underscore or separator, then the sample name.
    Example: '001_sampleA_1.dat' -> 'sampleA_1'
    """
    # Remove extension
    name, ext = os.path.splitext(filename)
    
    # Try Regex: match leading digits and optional separator, capture the rest
    match = re.match(r'^[\d]+[_\-]*(.*)$', name)
    if match and match.group(1):
        return match.group(1)
        
    return name # Fallback to full name without extension


def load_and_parse_directory(directory):
    """
    Finds all valid SAXS data files in the given directory and creates a
    dictionary representing rows in the interactive file table.
    """
    if not os.path.isdir(directory):
        return []

    extensions = ["*.dat", "*.csv", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    files = sorted([os.path.basename(f) for f in set(files)])
    
    file_data = []
    for f in files:
        bucket = parse_filename(f)
        file_data.append({
            "Move": True,
            "Filename": f,
            "Target Directory": bucket
        })
    return file_data

st.title("SAXS File Splitter")

# Sidebar
with st.sidebar:
    st.header("Directory Selection")
    
    # Initialize session state for working dir if not present
    if "splitter_dir" not in st.session_state:
        default_dir = os.getcwd()
        try:
            if os.path.exists("saxs_averager_state.json"):
                with open("saxs_averager_state.json", 'r') as f:
                    saved_state = json.load(f)
                    if "working_dir" in saved_state:
                        default_dir = saved_state["working_dir"]
        except Exception:
            pass
        st.session_state.splitter_dir = default_dir

    def clear_file_table():
        if "file_table" in st.session_state:
            del st.session_state["file_table"]

    if st.button("Browse for directory"):
        selected_dir = select_folder()
        if selected_dir:
            st.session_state.splitter_dir = selected_dir
            clear_file_table()
            st.rerun()

    st.text_input("Directory Path", key="splitter_dir", on_change=clear_file_table)
        
    st.markdown("---")
    if st.button("X Close Splitter"):
        close_window()


if not os.path.isdir(st.session_state.splitter_dir):
    st.warning("Please select a valid directory.")
    st.stop()
    
# Core Logic
if "file_table" not in st.session_state:
    raw_files = load_and_parse_directory(st.session_state.splitter_dir)
    if raw_files:
        st.session_state.file_table = pd.DataFrame(raw_files)
    else:
        st.session_state.file_table = pd.DataFrame(columns=["Move", "Filename", "Target Directory"])

df = st.session_state.file_table

if df.empty:
    st.info("No valid data files (.dat, .csv, .txt) found in the selected directory.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive File Assignment")
    st.markdown("Editable Table: Change the checkmarks to select files, and edit the **Target Directory** name to group files into different folders.")

    edited_df = st.data_editor(
        df,
        column_config={
            "Move": st.column_config.CheckboxColumn("Move?", required=True),
            "Filename": st.column_config.TextColumn("Filename", disabled=True),
            "Target Directory": st.column_config.TextColumn("Target Directory", required=True)
        },
        width='stretch',
        hide_index=True,
        height=500
    )
    
with col2:
    st.subheader("Summary")
    st.markdown("Planned Actions:")
    
    # Only consider files checked to move
    files_to_move = edited_df[edited_df["Move"]]
    if not files_to_move.empty:
        summary = files_to_move.groupby("Target Directory").size().reset_index(name='Count')
        st.dataframe(summary, width='stretch', hide_index=True)
    else:
        st.info("No files selected to move.")
        
    st.markdown("---")
    execute_bt = st.button("Split Selected Files", type="primary", width='stretch')

if execute_bt:
    if files_to_move.empty:
        st.error("No files selected to move.")
    else:
        # Execution loop
        success_count: int = 0
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(files_to_move)
        
        for i, (idx, row) in enumerate(files_to_move.iterrows()):
            filename = row["Filename"]
            target_dir_name = row["Target Directory"].strip() # Clean user input
            
            if not target_dir_name:
                 target_dir_name = "Uncategorized"
            
            source_path = os.path.join(st.session_state.splitter_dir, filename)
            target_path_dir = os.path.join(st.session_state.splitter_dir, target_dir_name)
            target_path_file = os.path.join(target_path_dir, filename)
            
            try:
                if not os.path.exists(target_path_dir):
                    os.makedirs(target_path_dir)
                shutil.move(source_path, target_path_file)
                success_count += 1
            except Exception as e:
                errors.append(f"Failed to move {filename}: {e}")
                
            progress_bar.progress((i + 1) / total_files)
            status_text.text(f"Processed {i + 1} of {total_files} files...")
            
        progress_bar.empty()
        status_text.empty()
        
        if success_count > 0:
            st.success(f"Successfully moved {success_count} files into their respective directories.")
            
            # Reload state
            st.session_state.file_table = pd.DataFrame(load_and_parse_directory(st.session_state.splitter_dir))
            st.rerun()
            
        if errors:
            st.error("Some files failed to move:")
            for err in errors:
                st.write(err)
