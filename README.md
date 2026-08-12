
# Dynamic resting-state functional connectivity: ENIGMA PGC PTSD


## `dfc_atlas_wf.py` — Atlas-based Dynamic Functional Connectivity (DFC)

Computes atlas-based **dynamic functional connectivity** from resting-state fMRI preprocessed with **HALFpipe** (ENIGMA/PGC pipeline). Uses a sliding-window approach: region-to-region correlations are recomputed across successive time windows, and DFC is defined as the **standard deviation of each connection's correlation across windows**. Runs subjects in parallel and can sweep multiple window/overlap combinations in one call.

**Input** (HALFpipe outputs):
- Parcel timeseries per subject — headerless TSV/CSV, rows = timepoints, columns = atlas regions (e.g. `*_atlas-schaefer2011Combined_timeseries.tsv`), under a `<site>/<subject>/` layout.
- Matching `.json` sidecar per file, providing `RepetitionTime` (per-subject TR read automatically; files without it are skipped).

**Output:**
- One region × region **DFC matrix per subject**, saved as headerless CSV.
- Organized as `dfc_<window>_<overlap>/<site>/<name>_DFC<window>_<overlap>.csv`; a separate tree per parameter combination.

**Key parameters:** `window_length_sec_list`, `overlap_percent_list`, `run_all_combinations` (cross-product vs. element-wise pairing), `discard_timepoints`, `num_processes`.

```python
dfc = DFC_Atlas(num_processes=4)
dfc.dfc_sliding_window_all(
    ts_dir=".../atlas_conn",
    ts_string="_atlas-schaefer2011Combined_timeseries",
    window_length_sec_list=[40, 50],
    overlap_percent_list=[25, 50, 75],
    output_dir=".../atlas_DFC",
    discard_timepoints=0,
    run_all_combinations=True,
)
```
