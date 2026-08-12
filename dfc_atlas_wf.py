import os
import json
import pandas as pd
import numpy as np
import warnings
import time
from functools import wraps
import multiprocessing as mp
from itertools import product
import re

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"-Class/Function {func.__name__}, time elapsed: {end_time - start_time:.3f} secs\n")
        return result
    return wrapper

def list_files_in_directory(folder_path, search_string):
    """
    List all files in a directory that contain a specific string in their filenames.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder to search for files.
    search_string : str
        String that should be present in the filenames.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if search_string in file:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    df = pd.DataFrame(file_paths, columns=['file_path'])
    return df

def add_tr_column(df, file_path_column='file_path'):
    """
    Add TR column by reading RepetitionTime from corresponding JSON files.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing file paths
    file_path_column : str, default='file_path'
        Name of column containing file paths
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new 'TR' column added
    """
    def get_tr_from_json(file_path):
        try:
            # Replace .tsv or .csv with .json
            json_path = file_path.replace('.tsv', '.json').replace('.csv', '.json')
            
            # Check if JSON file exists
            if not os.path.exists(json_path):
                return None
                
            # Read JSON and extract RepetitionTime
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('RepetitionTime', None)
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
    
    # Create a copy to avoid modifying original DataFrame
    df_result = df.copy()
    
    # Apply function to extract TR values
    df_result['TR'] = df_result[file_path_column].apply(get_tr_from_json)
    
    return df_result

def fc(ts, discard_timepoints=0, start_timepoint=None, end_timepoint=None, 
       interpolation_method='linear', min_valid_timepoints=3, 
       use_numpy=False, correlation_method='pearson'):
    """
    Calculate correlation matrix from rs-fMRI timeseries data with optimizations.
    
    Parameters:
    -----------
    ts : str or pd.DataFrame
        Path to CSV/TSV file (no header/index, rows=timepoints, cols=regions) or DataFrame
    discard_timepoints : int, default=0
        Number of initial timepoints to discard
    start_timepoint : int, optional
        Starting timepoint for correlation (after discarding, 0-indexed). Default: first timepoint
    end_timepoint : int, optional
        Ending timepoint for correlation (after discarding, exclusive). Default: last timepoint
    interpolation_method : str, default='linear'
        Method for interpolating missing values ('linear', 'nearest', 'cubic', None for no interpolation)
    min_valid_timepoints : int, default=3
        Minimum number of valid overlapping (between two brain regions) timepoints required for correlation
    use_numpy : bool, default=False
        Use numpy corrcoef instead of pandas corr for potentially faster computation
    correlation_method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall') - only used when use_numpy=False
    
    Returns:
    --------
    np.ndarray
        Correlation matrix between brain regions (NaN for removed/invalid regions)
    """
    # Print arguments
    print("FC Calculation Parameters:")
    print(f"Discard timepoints: {discard_timepoints}, "
            f"Start timepoint: {start_timepoint}, "
            f"End timepoint: {end_timepoint}, "
            f"Interpolation method: {interpolation_method}, "
            f"Min valid timepoints: {min_valid_timepoints}, "
            f"Correlation method: {correlation_method}, "
            f"Use numpy: {use_numpy}")    
    
    # Load data if ts is a string path, otherwise use DataFrame directly
    if isinstance(ts, str):
        # Determine file type based on extension
        if ts.lower().endswith('.tsv'):
            data = pd.read_csv(ts, sep='\t', header=None, index_col=None)
        elif ts.lower().endswith('.csv'):
            data = pd.read_csv(ts, sep=',', header=None, index_col=None)
        else:
            # Try to infer separator by reading first few lines
            try:
                # First try comma separator
                data = pd.read_csv(ts, sep=',', header=None, index_col=None)
                # Check if all data is in first column (suggests wrong separator)
                if data.shape[1] == 1:
                    # Try tab separator
                    data = pd.read_csv(ts, sep='\t', header=None, index_col=None)
            except:
                # Fallback: let pandas infer the separator
                data = pd.read_csv(ts, sep=None, engine='python', header=None, index_col=None)
    else:
        data = ts.copy()  # Work with a copy to avoid modifying original
    
    # CRITICAL FIX: Convert all columns to numeric, coercing errors to NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Check if we have any numeric data left
    if data.select_dtypes(include=[np.number]).empty:
        print(f"Warning: No numeric data found in {ts}")
        n_cols = data.shape[1]
        return np.full((n_cols, n_cols), np.nan)
    
    # Discard initial timepoints
    if discard_timepoints > 0:
        data = data.iloc[discard_timepoints:]
    
    # Apply start and end timepoint selection
    start_idx = start_timepoint if start_timepoint is not None else 0
    end_idx = end_timepoint if end_timepoint is not None else len(data)
    data = data.iloc[start_idx:end_idx]
    
    # Check if we have enough timepoints
    if len(data) < min_valid_timepoints:
        n_cols = data.shape[1]
        return np.full((n_cols, n_cols), np.nan)
    
    # Store original column indices and identify all-NaN columns
    original_cols = data.columns
    all_nan_cols = data.columns[data.isna().all()]
    
    # Early exit if all columns are NaN
    if len(all_nan_cols) == len(original_cols):
        n_cols = len(original_cols)
        return np.full((n_cols, n_cols), np.nan)
    
    # Remove columns with all NaN values for correlation calculation
    valid_data = data.dropna(axis=1, how='all')
    
    # Handle missing values based on interpolation method
    if interpolation_method is not None and not valid_data.empty:
        # Only interpolate if we have numeric columns
        numeric_cols = valid_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            valid_data = valid_data[numeric_cols].interpolate(
                method=interpolation_method, axis=0, limit_direction='both'
            )
    
    # Check for columns with insufficient valid data after interpolation
    if interpolation_method is None:
        # Count valid timepoints per column
        valid_counts = valid_data.count()
        insufficient_cols = valid_counts < min_valid_timepoints
        valid_data.loc[:, insufficient_cols] = np.nan
    
    # Choose computation method based on use_numpy flag and data characteristics
    n_regions = valid_data.shape[1]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if use_numpy and n_regions > 0:
            # NumPy approach - faster for large matrices with clean data
            try:
                # Remove any remaining rows with all NaN
                clean_data = valid_data.dropna(how='all')
                if len(clean_data) >= min_valid_timepoints:
                    # Use numpy's corrcoef for potentially faster computation
                    valid_corr = pd.DataFrame(
                        np.corrcoef(clean_data.T),
                        index=valid_data.columns,
                        columns=valid_data.columns
                    )
                else:
                    # Not enough data points
                    valid_corr = pd.DataFrame(
                        np.full((n_regions, n_regions), np.nan),
                        index=valid_data.columns,
                        columns=valid_data.columns
                    )
            except:
                # Fallback to pandas if numpy fails
                valid_corr = valid_data.corr(method=correlation_method, min_periods=min_valid_timepoints)
        else:
            # Pandas approach - more robust for missing data
            valid_corr = valid_data.corr(method=correlation_method, min_periods=min_valid_timepoints)
    
    # Create full correlation matrix with NaN for removed regions
    full_corr = pd.DataFrame(index=original_cols, columns=original_cols, dtype=float)
    
    # Fill in the valid correlations at their original positions
    if not valid_corr.empty:
        full_corr.loc[valid_corr.index, valid_corr.columns] = valid_corr
    
    # # Print part of the matrix for verification
    # print("Correlation matrix (partial view):")
    # print(full_corr.iloc[:5, :5])
    
    return full_corr.values

def fisher_z_transform(correlations):
    """
    Apply Fisher Z-transformation to correlation coefficients.
    
    Parameters:
    -----------
    correlations : array-like
        Correlation coefficients
        
    Returns:
    --------
    z_scores : array
        Fisher Z-transformed values
    """
    # Clip correlations to avoid numerical issues
    correlations = np.clip(correlations, -0.999999, 0.999999)
    # Use arctanh (equivalent to Fisher Z) following nilearn example
    z_scores = np.arctanh(correlations)
    return z_scores

@timeit
def dfc_sliding_window(args):
    """
    Calculate atlas_based dynamic functional connectivity (DFC).
    
    Parameters:
    -----------
    args : tuple
        Contains all necessary parameters for DFC calculation
        (ts, tr, window_length_sec, overlap_percent, output_path, discard_timepoints)
        
    Returns:
    --------
    None (saves output to file)
    """
    (ts, tr, window_length_sec, overlap_percent, output_path, discard_timepoints) = args
    
    print(f"DFC Processing: {os.path.basename(ts)}")
    print(f"Window: {window_length_sec}s, Overlap: {overlap_percent}%")
    
    # Convert window parameters
    window_length_tp = int(np.round(window_length_sec / tr))
    overlap_tp = int(np.round(window_length_tp * overlap_percent / 100))
    step_size = window_length_tp - overlap_tp
    
    print(f"Window: {window_length_tp} TRs ({window_length_sec}s), "
          f"Overlap: {overlap_tp} TRs ({overlap_percent}%), "
          f"Step: {step_size} TRs")
    
    try:
        # Load data - only handle CSV/TSV files
        if isinstance(ts, str):
            # Determine file type based on extension
            if ts.lower().endswith('.tsv'):
                data = pd.read_csv(ts, sep='\t', header=None, index_col=None)
            elif ts.lower().endswith('.csv'):
                data = pd.read_csv(ts, sep=',', header=None, index_col=None)
            else:
                # Try to infer separator for other text files
                try:
                    # First try comma separator
                    data = pd.read_csv(ts, sep=',', header=None, index_col=None)
                    # Check if all data is in first column (suggests wrong separator)
                    if data.shape[1] == 1:
                        # Try tab separator
                        data = pd.read_csv(ts, sep='\t', header=None, index_col=None)
                except:
                    # Fallback: let pandas infer the separator
                    data = pd.read_csv(ts, sep=None, engine='python', header=None, index_col=None)
        else:
            # Handle DataFrame input
            data = ts.copy()  # Work with a copy to avoid modifying original

        # Convert all columns to numeric, coercing errors to NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Check if we have any numeric data left
        if data.select_dtypes(include=[np.number]).empty:
            print(f"Warning: No numeric data found in {ts}. Skipping this file.")
            return  # Skip this file
            
        # Discard initial timepoints
        if discard_timepoints > 0:
            data = data.iloc[discard_timepoints:]   
               
        n_timepoints, n_regions = data.shape
        print(f"Processing {n_regions} regions with {n_timepoints} timepoints")
        
        # Calculate number of windows
        n_windows = (n_timepoints - window_length_tp) // step_size + 1
        if n_windows < 2:
            print(f"Warning: Insufficient timepoints for DFC calculation. Skipping {ts}")
            return
        
        print(f"Computing DFC across {n_windows} windows...")
        
        # Calculate the start and end timepoints for each window
        window_start_end = [(i * step_size, i * step_size + window_length_tp) for i in range(n_windows)]    
        
        # Calculate sliding window correlations
        all_window_correlations = []
        for start_tp, end_tp in window_start_end:
            corr_matrix = fc(data, discard_timepoints=0, start_timepoint=start_tp, end_timepoint=end_tp, 
                             interpolation_method='linear', min_valid_timepoints=3, 
                             use_numpy=False, correlation_method='pearson')
            all_window_correlations.append(corr_matrix)
        correlations = np.array(all_window_correlations)  # Shape: (n_windows, n_regions, n_regions)
        
        # Calculate DFC as standard deviation of correlations across windows
        dfc_values = np.nanstd(correlations, axis=0) # Using nanstd to handle NaN values
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save output
        pd.DataFrame(dfc_values).to_csv(output_path, header=False, index=False, na_rep='nan')
        
        print(f"DFC calculation completed!")
        print(f"DFC range: {np.nanmin(dfc_values):.6f} to {np.nanmax(dfc_values):.6f}")
        print(f"Mean DFC: {np.nanmean(dfc_values):.6f}")
        print(f"Output saved to: {output_path}")
        
        return pd.DataFrame(dfc_values)
        
    except Exception as e:
        print(f"Error processing {ts}: {str(e)}")
        print(f"Skipping this file and continuing...")
        return  # Skip this file instead of raising the error

@timeit
class DFC_Atlas:
    """
    Class for Dynamic Functional Connectivity analysis using atlas_based timeseries.
    """
    
    def __init__(self, num_processes=None):
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        
    def dfc_sliding_window_all(self, ts_dir, ts_string, window_length_sec_list, overlap_percent_list, 
                               output_dir, discard_timepoints, run_all_combinations=True):
        """
        Calculate DFC for multiple subjects and parameter combinations in parallel.
        
        Parameters:
        -----------
        ts_dir : str
            Directory containing TSV/CSV files of timeseries files (rows=timepoints, cols=regions)
        ts_string : str
            String to identify relevant timeseries files in the directory
        window_length_sec_list : list or float
            List of window lengths in seconds, or single value
        overlap_percent_list : list or float  
            List of overlap percentages, or single value
        output_dir : str
            Base directory to save output DFC files
        discard_timepoints : int
            Number of initial timepoints to discard from each timeseries
        run_all_combinations : bool, default=True
            If True, run all combinations of window_length_sec and overlap_percent
            If False, pair them element-wise (lists must be same length)
        """
        print(f"\nStarting batch DFC calculation...")
        print(f"Using {self.num_processes} processes")
        
        # Convert single values to lists
        if not isinstance(window_length_sec_list, list):
            window_length_sec_list = [window_length_sec_list]
        if not isinstance(overlap_percent_list, list):
            overlap_percent_list = [overlap_percent_list]
            
        # Generate parameter combinations
        if run_all_combinations:
            param_combinations = list(product(window_length_sec_list, overlap_percent_list))
            print(f"Running all combinations: {len(param_combinations)} parameter sets")
        else:
            if len(window_length_sec_list) != len(overlap_percent_list):
                raise ValueError("When run_all_combinations=False, window_length_sec_list and overlap_percent_list must have the same length")
            param_combinations = list(zip(window_length_sec_list, overlap_percent_list))
            print(f"Running paired combinations: {len(param_combinations)} parameter sets")
            
        print("Parameter combinations:")
        for i, (win_len, overlap) in enumerate(param_combinations):
            print(f"  {i+1}. Window: {win_len}s, Overlap: {overlap}%")
        
        # Find all relevant timeseries files
        df = list_files_in_directory(ts_dir, ts_string)
    
        # Add subject-specific TR
        df = add_tr_column(df)
        
        # Filter out files without TR values (these may be problematic)
        df_valid = df.dropna(subset=['TR'])
        if len(df_valid) < len(df):
            print(f"Warning: {len(df) - len(df_valid)} files excluded due to missing TR values")
        df = df_valid
        
        print(f"Found {len(df)} valid timeseries files")
            
        # Prepare arguments for all combinations
        all_args = []
        
        for window_length_sec, overlap_percent in param_combinations:
            print(f"\nPreparing jobs for Window: {window_length_sec}s, Overlap: {overlap_percent}%")
            
            # Create parameter-specific output directory
            param_output_dir = os.path.join(output_dir, f"dfc_{window_length_sec}_{overlap_percent}")
            os.makedirs(param_output_dir, exist_ok=True)
            
            # Add output paths to DataFrame for this parameter combination
            for _, row in df.iterrows():
                sitename = os.path.basename(os.path.dirname(os.path.dirname(row['file_path'])))
                file_name = os.path.basename(row['file_path'])
                base_name = re.sub(r'\.(tsv|csv)$', f'_DFC{window_length_sec}_{overlap_percent}.csv', file_name)
                out_path = os.path.join(param_output_dir, sitename, base_name)
                
                # Prepare arguments: (ts, tr, window_length_sec, overlap_percent, output_path, discard_timepoints)
                args = (
                    row['file_path'],
                    row['TR'],
                    window_length_sec,
                    overlap_percent,
                    out_path,
                    discard_timepoints
                )
                all_args.append(args)
        
        print(f"\nTotal jobs to process: {len(all_args)}")
        print(f"Jobs per parameter combination: {len(df)}")
        
        # Single job for debugging (uncomment if needed)
        # print("Running single job for debugging...")
        # dfc_sliding_window(all_args[0])
        # return

        # Process all combinations in parallel
        print("Starting parallel processing...")
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(dfc_sliding_window, all_args)
        
        print(f"\nCompleted processing:")
        print(f"  - {len(param_combinations)} parameter combinations")
        print(f"  - {len(df)} subjects per combination")
        print(f"  - {len(all_args)} total jobs processed")
        
        # Summary of outputs
        print(f"\nOutput directories created:")
        for window_length_sec, overlap_percent in param_combinations:
            param_dir = os.path.join(output_dir, f"dfc_{window_length_sec}_{overlap_percent}")
            print(f"  - {param_dir}")


if __name__ == "__main__":
    
    # # Example: dfc_sliding_window
    # ts = "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/aCompCor/atlas_conn/AMC/sub-1132/sub-1132_task-rest_feature-corrMatrix_atlas-schaefer2011Combined_timeseries.tsv"
    # tr = 2
    # window_length_sec = 15
    # overlap_percent = 20
    # output_path = "/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/my_data/atlas_DFC/test_dfc.csv"
    # discard_timepoints = 0
    # my_args = (ts, tr, window_length_sec, overlap_percent, output_path, discard_timepoints)
    # df = dfc_sliding_window(my_args)
    # pass
    
    
    
    
    # Example: DFC_Atlas, with multiple parameter combinations
    
    # Define parameter combinations to test
    window_length_sec_list = [40, 50] #[20, 30, 60, 90]  # Different window lengths in secs
    overlap_percent_list = [25, 50, 75]    # Different overlap percentages (%)
    
    # Initialize DFC processor
    dfc = DFC_Atlas(num_processes=4)
    
    # Option 1: Run all combinations
    dfc.dfc_sliding_window_all(
        ts_dir="/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/aCompCor/atlas_conn",
        ts_string="_task-rest_feature-corrMatrix_atlas-schaefer2011Combined_timeseries",
        window_length_sec_list=window_length_sec_list,   
        overlap_percent_list=overlap_percent_list,
        output_dir="/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/my_data/atlas_DFC",
        discard_timepoints=0,
        run_all_combinations=True  # This will make different output folders based on all combinations of window_length_sec_list and overlap_percent_list
    )
    
    # Option 2: Run paired combinations (comment out Option 1 to use this)
    # window_length_sec_list = [30, 45, 60]  # Different window lengths in secs
    # overlap_percent_list = [25, 50, 75]    # Different overlap percentages (%)
    # This would run only 3 combinations: (30,25), (45,50), (60,75)
    # dfc.dfc_sliding_window_all(
    #     ts_dir="/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/aCompCor/atlas_conn",
    #     ts_string="_task-rest_feature-corrMatrix_atlas-schaefer2011Combined_timeseries",
    #     window_length_sec_list=window_length_sec_list,   
    #     overlap_percent_list=overlap_percent_list,
    #     output_dir="/mnt/munin/Morey/Lab/Delin/Projects/IBMMA/Data/my_data/atlas_DFC",
    #     discard_timepoints=0,
    #     run_all_combinations=False  # This will create 3 different output folders
    # )
    
    pass