import tkinter as tk
from tkinter import filedialog
import csv
import uuid
import random
from datetime import datetime, timedelta

def adjust_timestamps():
    """Reads the chosen CSV file, modifies timestamps based on the user-supplied start time, and
    saves a new CSV on the Desktop with a random filename."""
    
    # Get user input for new starting timestamp
    new_start_str = entry_timestamp.get().strip()
    if not new_start_str:
        status_label.config(text="Please enter a valid start timestamp.")
        return
    
    # Attempt parsing the user-supplied date/time
    try:
        new_start_dt = datetime.strptime(new_start_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # If the user forgot fractional seconds, try again without them
        try:
            new_start_dt = datetime.strptime(new_start_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            status_label.config(text="Invalid date/time format. Try: 2025-01-21 07:16:05.932804")
            return

    if not csv_file_path.get():
        status_label.config(text="Please select a CSV file first.")
        return

    file_path = csv_file_path.get()

    # Read the CSV data
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        status_label.config(text="CSV file contains no data rows.")
        return

    # Header row should be the first row
    header = rows[0]
    # Data rows
    data_rows = rows[1:]

    # Identify which column is the timestamp (assumes the column is named 'timestamp')
    try:
        timestamp_index = header.index("timestamp")
    except ValueError:
        status_label.config(text="No 'timestamp' column found in the CSV header.")
        return
# Parse original timestamps and compute intervals
    original_datetimes = []
    for row in data_rows:
        ts_str = row[timestamp_index]
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            # If missing fractional seconds, parse again
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        original_datetimes.append(dt)
    
    # Build a list of intervals between consecutive rows
    intervals = []
    for i in range(1, len(original_datetimes)):
        intervals.append(original_datetimes[i] - original_datetimes[i - 1])

    # Create a new list to store updated timestamps
    new_datetimes = []
    # The first row takes the user-specified new start time
    if original_datetimes:
        new_datetimes.append(new_start_dt)

    # If you want strictly the same intervals, use intervals[i - 1] 
    # If you want a small random jitter around the existing interval, 
    # you can add something like "± 0.1 second random"
    # interval_jitter_seconds = 0.1
    for i in range(1, len(original_datetimes)):
        # Strictly same interval, or add a small random jitter:
        # jitter = random.uniform(-interval_jitter_seconds, interval_jitter_seconds)
        # new_time = new_datetimes[i-1] + intervals[i - 1] + timedelta(seconds=jitter)
        new_time = new_datetimes[i - 1] + intervals[i - 1]
        new_datetimes.append(new_time)

    # Apply these new timestamps to the data rows
    for i, row in enumerate(data_rows):
        # Convert the updated datetime to string
        # Keep the same microsecond precision
        if i < len(new_datetimes):
            row[timestamp_index] = new_datetimes[i].strftime("%Y-%m-%d %H:%M:%S.%f")
        else:
            # If something goes out of range, keep original
            pass
# Generate random filename
    random_filename = f"{uuid.uuid4()}.csv"
    save_path = r"C:\Users\Mohamed\Desktop"  # <--- change this path if needed
    output_csv = f"{save_path}\\{random_filename}"

    # Write the new CSV to Desktop
    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        # Write header
        writer.writerow(header)
        # Write modified data rows
        writer.writerows(data_rows)

    status_label.config(text=f"New CSV saved as: {output_csv}")

def browse_file():
    """Open a file dialog for selecting the CSV file."""
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=(("CSV files", ".csv"), ("All files", ".*"))
    )
    if file_path:
        csv_file_path.set(file_path)
        status_label.config(text=f"Selected file: {file_path}")

# ----------------------- GUI Setup -----------------------
root = tk.Tk()
root.title("Timestamp Adjuster")

# Frame for file selection
frame_file = tk.Frame(root)
frame_file.pack(padx=10, pady=5, fill='x')

btn_browse = tk.Button(frame_file, text="Browse CSV", command=browse_file)
btn_browse.pack(side=tk.LEFT)

csv_file_path = tk.StringVar()
entry_file = tk.Entry(frame_file, textvariable=csv_file_path, width=60, state='readonly')
entry_file.pack(side=tk.LEFT, padx=5)

# Frame for timestamp entry
frame_ts = tk.Frame(root)
frame_ts.pack(padx=10, pady=5, fill='x')

lbl_timestamp = tk.Label(frame_ts, text="New Start Timestamp:")
lbl_timestamp.pack(side=tk.LEFT)

entry_timestamp = tk.Entry(frame_ts, width=30)
entry_timestamp.insert(0, "2025-01-21 07:16:05.932804")  # default example
entry_timestamp.pack(side=tk.LEFT, padx=5)

# Button to start processing
btn_start = tk.Button(root, text="Adjust Timestamps", command=adjust_timestamps)
btn_start.pack(pady=10)

# Status label
status_label = tk.Label(root, text="", fg="blue")
status_label.pack(pady=5)

root.mainloop()
