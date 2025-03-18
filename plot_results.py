#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_mpi_speedup():
    """
    Reads 'data/sequential_baseline.csv' for the baseline runtime and
    'data/domain_decomposition_results.csv' for MPI domain decomposition results.
    Computes average TotalTime per process count and calculates speedup.
    """
    seq_df = pd.read_csv("data/sequential_baseline.csv")
    # Use the sequential time for a chosen image (assumed first) as baseline
    baseline_time = seq_df.loc[0, "TotalTime"]
    
    mpi_domain_df = pd.read_csv("data/domain_decomposition_results.csv")
    mpi_avg = mpi_domain_df.groupby("NumProcs")["TotalTime"].mean().reset_index()
    mpi_avg["Speedup"] = baseline_time / mpi_avg["TotalTime"]
    
    # Convert to numpy arrays
    num_procs = mpi_avg["NumProcs"].to_numpy()
    speedup = mpi_avg["Speedup"].to_numpy()
    
    plt.figure(figsize=(8,6))
    plt.plot(num_procs, speedup, marker='o', label="MPI Domain")
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup (Sequential Time / Parallel Time)")
    plt.title("MPI Speedup vs. Number of Processes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/mpi_speedup.png")
    plt.close()

# ------------------------- Graph 9: OpenMP Speedup vs. Number of Threads -------------------------
def plot_openmp_speedup():
    """
    Reads 'data/sequential_baseline.csv' for the baseline runtime and
    'data/openmp_scaling.csv' for OpenMP results.
    Groups by thread count, computes the average TotalTime,
    and then calculates speedup as baseline_time / openmp_time.
    """
    try:
        seq_df = pd.read_csv("data/sequential_baseline.csv")
        omp_df = pd.read_csv("data/openmp_scaling.csv")
    except Exception as e:
        print("Required CSV file not found:", e)
        return
    
    baseline_time = seq_df.loc[0, "TotalTime"]
    omp_avg = omp_df.groupby("Threads")["TotalTime"].mean().reset_index()
    omp_avg["Speedup"] = baseline_time / omp_avg["TotalTime"]
    
    # Convert series to numpy arrays
    threads = omp_avg["Threads"].to_numpy()
    speedup = omp_avg["Speedup"].to_numpy()
    
    plt.figure(figsize=(8,6))
    plt.plot(threads, speedup, marker='o', label="OpenMP")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (Sequential Time / OpenMP Time)")
    plt.title("OpenMP Speedup vs. Number of Threads")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/openmp_speedup.png")
    plt.close()

# ------------------------- Graph 10: Combined Scaling Comparison -------------------------
def plot_combined_scaling():
    """
    Creates a combined figure with two subplots:
    one for OpenMP scaling (speedup vs. threads) and one for MPI scaling (speedup vs. processes).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    
    # OpenMP scaling subplot
    try:
        seq_df = pd.read_csv("data/sequential_baseline.csv")
        omp_df = pd.read_csv("data/openmp_scaling.csv")
    except Exception as e:
        print("OpenMP scaling CSV file not found:", e)
        return
    baseline_time = seq_df.loc[0, "TotalTime"]
    omp_avg = omp_df.groupby("Threads")["TotalTime"].mean().reset_index()
    omp_avg["Speedup"] = baseline_time / omp_avg["TotalTime"]
    ax1.plot(omp_avg["Threads"].to_numpy(), omp_avg["Speedup"].to_numpy(), marker='o', label="OpenMP")
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Speedup")
    ax1.set_title("OpenMP Scaling")
    ax1.grid(True)
    
    try:
        mpi_df = pd.read_csv("data/domain_decomposition_results.csv")
    except Exception as e:
        print("MPI CSV file not found:", e)
        return
    mpi_avg = mpi_df.groupby("NumProcs")["TotalTime"].mean().reset_index()
    mpi_avg["Speedup"] = baseline_time / mpi_avg["TotalTime"]
    ax2.plot(mpi_avg["NumProcs"].to_numpy(), mpi_avg["Speedup"].to_numpy(), marker='o', label="MPI Domain")
    ax2.set_xlabel("Number of Processes")
    ax2.set_ylabel("Speedup")
    ax2.set_title("MPI Scaling")
    ax2.grid(True)
    
    plt.suptitle("Combined Parallel Scaling Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("graphs/combined_scaling.png")
    plt.close()

# ------------------------- Graph 3: Weak Scaling Efficiency -------------------------



def plot_weak_scaling_efficiencies():
    """
    Creates a 1x2 subplot figure for weak scaling efficiency:
     - Left subplot: Efficiency vs. Number of Processes for increasing frames
     - Right subplot: Efficiency vs. Number of Processes for increasing resolution
    
    Reads:
      data/weak_scaling_frames.csv -> columns: NumProcs, NumFrames, TotalTime
      data/weak_scaling_size.csv   -> columns: NumProcs, Resolution, TotalTime
    
    Efficiency is computed as T(1) / T(N), where T(1) is the average runtime at 1 process.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    
    # ------------------ Subplot 1: Weak Scaling (Increasing Frames) ------------------
    try:
        df_frames = pd.read_csv("data/weak_scaling_frames.csv")
    except Exception as e:
        print("data/weak_scaling_frames.csv not found or invalid.")
        return
    
    # Convert columns to numeric
    df_frames["NumProcs"]  = pd.to_numeric(df_frames["NumProcs"],  errors="coerce")
    df_frames["TotalTime"] = pd.to_numeric(df_frames["TotalTime"], errors="coerce")
    
    # Baseline time at NumProcs=1
    baseline_frames = df_frames[df_frames["NumProcs"] == 1]["TotalTime"].mean()
    df_frames["Efficiency"] = baseline_frames / df_frames["TotalTime"]
    
    # Sort by NumProcs just in case
    df_frames = df_frames.sort_values(by="NumProcs")
    
    # Convert Series to numpy arrays
    x_frames = df_frames["NumProcs"].to_numpy()
    y_frames = df_frames["Efficiency"].to_numpy()

    ax1.plot(x_frames, y_frames, marker='o')
    ax1.set_xlabel("Number of Processes")
    ax1.set_ylabel("Weak Scaling Efficiency")
    ax1.set_title("MPI Weak Scaling Efficiency (Increasing Frames)")
    ax1.grid(True)
    
    # ------------------ Subplot 2: Weak Scaling (Increasing Resolution) ------------------
    try:
        df_size = pd.read_csv("data/weak_scaling_size.csv")
    except Exception as e:
        print("data/weak_scaling_size.csv not found or invalid.")
        return
    
    df_size["NumProcs"]    = pd.to_numeric(df_size["NumProcs"],    errors="coerce")
    df_size["TotalTime"]   = pd.to_numeric(df_size["TotalTime"],   errors="coerce")
    
    baseline_size = df_size[df_size["NumProcs"] == 1]["TotalTime"].mean()
    df_size["Efficiency"] = baseline_size / df_size["TotalTime"]
    
    # Sort by NumProcs for consistency
    df_size = df_size.sort_values(by="NumProcs")

    x_size = df_size["NumProcs"].to_numpy()
    y_size = df_size["Efficiency"].to_numpy()

    ax2.plot(x_size, y_size, marker='o')
    ax2.set_xlabel("Number of Processes")
    ax2.set_ylabel("Weak Scaling Efficiency")
    ax2.set_title("MPI Weak Scaling Efficiency (Increasing Resolution)")
    ax2.grid(True)
    
    # Final layout and save
    plt.tight_layout()
    plt.savefig("graphs/weak_scaling_efficiencies_combined.png")
    plt.close()


def plot_openmp_weak_scaling_efficiencies():
    """
    Creates a 1x2 subplot figure for OpenMP weak scaling efficiency:
     - Left subplot: Efficiency vs. Threads for increasing frames
     - Right subplot: Efficiency vs. Threads for increasing resolution

    Reads:
      data/openmp_weak_scaling_frames.csv -> columns: Threads, NumFrames, TotalTime
      data/openmp_weak_scaling_size.csv   -> columns: Threads, Resolution, TotalTime

    Efficiency = T(1-thread) / T(n-threads).
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ------------------ Subplot 1: Weak Scaling Efficiency (Increasing Frames) ------------------
    try:
        df_frames = pd.read_csv("data/openmp_weak_scaling_frames.csv")
    except Exception as e:
        print("data/openmp_weak_scaling_frames.csv not found or invalid.")
        return

    df_frames["Threads"]   = pd.to_numeric(df_frames["Threads"],   errors="coerce")
    df_frames["TotalTime"] = pd.to_numeric(df_frames["TotalTime"], errors="coerce")

    baseline_frames = df_frames[df_frames["Threads"] == 1]["TotalTime"].mean()
    df_frames["Efficiency"] = baseline_frames / df_frames["TotalTime"]
    df_frames = df_frames.sort_values(by="Threads")

    x_fr = df_frames["Threads"].to_numpy()
    y_fr = df_frames["Efficiency"].to_numpy()

    ax1.plot(x_fr, y_fr, marker='o', color="#1f77b4")
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Weak Scaling Efficiency")
    ax1.set_title("OpenMP Weak Scaling (Increasing Frames)")
    ax1.grid(True)

    # ------------------ Subplot 2: Weak Scaling Efficiency (Increasing Resolution) ------------------
    try:
        df_size = pd.read_csv("data/openmp_weak_scaling_size.csv")
    except Exception as e:
        print("data/openmp_weak_scaling_size.csv not found or invalid.")
        return

    df_size["Threads"]   = pd.to_numeric(df_size["Threads"],   errors="coerce")
    df_size["TotalTime"] = pd.to_numeric(df_size["TotalTime"], errors="coerce")

    baseline_size = df_size[df_size["Threads"] == 1]["TotalTime"].mean()
    df_size["Efficiency"] = baseline_size / df_size["TotalTime"]
    df_size = df_size.sort_values(by="Threads")

    x_sz = df_size["Threads"].to_numpy()
    y_sz = df_size["Efficiency"].to_numpy()

    ax2.plot(x_sz, y_sz, marker='o', color="#ff7f0e")
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel("Weak Scaling Efficiency")
    ax2.set_title("OpenMP Weak Scaling (Increasing Resolution)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("graphs/openmp_weak_scaling_efficiencies_combined.png")
    plt.close()

# ------------------------- Graph 4: CUDA Time Breakdown -------------------------
def plot_cuda_time_breakdown():
    # Use the aggregated CUDA results file which has:
    # ImageName,LoadTime,FilterTime,ExportTime,TotalTime
    try:
        df = pd.read_csv("data/cuda_results.csv")
    except Exception as e:
        print("CUDA results CSV not found.")
        return

    labels = df["ImageName"]
    load_time = df["LoadTime"]
    filter_time = df["FilterTime"]
    export_time = df["ExportTime"]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x, load_time.values, width, label="Load Time")
    ax.bar(x, filter_time.values, width, bottom=load_time.values, label="Filter Time")
    ax.bar(x, export_time.values, width, bottom=load_time.values + filter_time.values, label="Export Time")
    
    ax.set_ylabel("Time (s)")
    ax.set_title("CUDA Time Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("graphs/cuda_time_breakdown.png")
    plt.close()

# ------------------------- Graph 5: Communication vs. Computation Overhead (MPI Domain) -------------------------
def plot_mpi_comm_vs_comp():
    # Assume a CSV "data/domain_decomposition_comm_comp.csv" with columns:
    # ImageName,CommTime,CompTime,TotalTime
    try:
        df = pd.read_csv("data/domain_decomposition_comm_comp.csv")
    except Exception as e:
        print("Domain decomposition comm/comp CSV not found.")
        return
    labels = df["ImageName"]
    comm = df["CommTime"]
    comp = df["CompTime"]
    
    x = np.arange(len(labels))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x, comm.values, width, label="Communication (Ghost Exchange)")
    ax.bar(x, comp.values, width, bottom=comm.values, label="Computation (Filtering)")
    
    ax.set_ylabel("Time (s)")
    ax.set_title("MPI Domain Decomposition: Communication vs. Computation Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("graphs/mpi_comm_vs_comp.png")
    plt.close()


# ------------------------- Graph 6: Runtime Comparison per GIF -------------------------
def plot_runtime_comparison():
    """
    Reads 'data/compare_all.csv' and creates a grouped bar chart comparing 
    only the Sequential, OpenMP, MPI Domain, and CUDA implementations on a log scale.

    CSV Columns Expected:
    Image,Sequential,OpenMP,MPI+OpenMP,Hybrid,CUDA,MPI Frames,MPI Domain
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Attempt to read the CSV file
    try:
        df = pd.read_csv("data/compare_all.csv")
    except Exception as e:
        print("compare_all.csv not found in data folder.")
        return
    
    # Restrict to the four versions we want to compare
    versions = ["Sequential", "OpenMP", "MPI Domain", "CUDA"]
    
    # Convert relevant columns to numeric
    for ver in versions:
        df[ver] = pd.to_numeric(df[ver], errors="coerce")
    
    images = df["Image"]
    n = len(images)
    ind = np.arange(n)  # x locations for the groups

    # Adjust bar width for four bars in each group
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Set up offsets for grouped bars: 4 versions => 4 offsets around the group center
    offsets = np.arange(len(versions)) * width - (len(versions) - 1) * width / 2

    # Custom colors for the 4 versions
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    
    # Plot each version's bars
    for i, ver in enumerate(versions):
        ax.bar(ind + offsets[i], df[ver], width, label=ver, color=colors[i])

    ax.set_xlabel("GIF Image Name")
    ax.set_ylabel("Runtime (s)")

    # Use a log scale on the y-axis so that small values remain visible
    ax.set_yscale('log')

    ax.set_title("Runtime Comparison per GIF (Log Scale)")
    ax.set_xticks(ind)
    ax.set_xticklabels(images, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("graphs/runtime_comparison.png")
    plt.close()


def plot_increasing_data():
    """
    Reads 'data/increasing_frames.csv' and 'data/increasing_size.csv',
    then creates a 1x2 subplot figure:
      - Left subplot: Runtime vs. Number of Frames
      - Right subplot: Runtime vs. Image Width
    Each subplot shows lines for Sequential, OpenMP, MPI Domain, and CUDA.
    """

    # --- Part 1: Read and plot 'increasing_frames.csv' ---
    try:
        frames_df = pd.read_csv("data/increasing_frames.csv")
    except Exception as e:
        print("data/increasing_frames.csv not found or invalid.")
        return

    # Convert columns to numeric
    frames_df["NumFrames"] = pd.to_numeric(frames_df["NumFrames"], errors="coerce")
    for col in ["Sequential", "OpenMP", "MPI_Domain", "CUDA"]:
        frames_df[col] = pd.to_numeric(frames_df[col], errors="coerce")

    # Sort by NumFrames ascending, in case it's out of order
    frames_df = frames_df.sort_values(by="NumFrames")

    num_frames = frames_df["NumFrames"].to_numpy()
    seq_time_f = frames_df["Sequential"].to_numpy()
    omp_time_f = frames_df["OpenMP"].to_numpy()
    mpi_time_f = frames_df["MPI_Domain"].to_numpy()
    cuda_time_f = frames_df["CUDA"].to_numpy()

    # --- Part 2: Read and plot 'increasing_size.csv' ---
    try:
        size_df = pd.read_csv("data/increasing_size.csv")
    except Exception as e:
        print("data/increasing_size.csv not found or invalid.")
        return

    for col in ["Sequential", "OpenMP", "MPI_Domain", "CUDA"]:
        size_df[col] = pd.to_numeric(size_df[col], errors="coerce")

    # Extract width from the Resolution column
    # Format is "WxH" (e.g., "100x100") => split and convert the first part to float
    size_df["Width"] = size_df["Resolution"].str.split("x", expand=True)[0].astype(float)
    size_df = size_df.sort_values(by="Width")  # ensure ascending order

    width = size_df["Width"].to_numpy()
    seq_time_s = size_df["Sequential"].to_numpy()
    omp_time_s = size_df["OpenMP"].to_numpy()
    mpi_time_s = size_df["MPI_Domain"].to_numpy()
    cuda_time_s = size_df["CUDA"].to_numpy()

    # --- Create a 1 x 2 figure with subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Runtime vs. Number of Frames
    ax1.plot(num_frames, seq_time_f, marker='o', label="Sequential")
    ax1.plot(num_frames, omp_time_f, marker='o', label="OpenMP")
    ax1.plot(num_frames, mpi_time_f, marker='o', label="MPI Domain")
    ax1.plot(num_frames, cuda_time_f, marker='o', label="CUDA")
    ax1.set_xlabel("Number of Frames")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime vs. Number of Frames")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Runtime vs. Image Width
    ax2.plot(width, seq_time_s, marker='o', label="Sequential")
    ax2.plot(width, omp_time_s, marker='o', label="OpenMP")
    ax2.plot(width, mpi_time_s, marker='o', label="MPI Domain")
    ax2.plot(width, cuda_time_s, marker='o', label="CUDA")
    ax2.set_xlabel("Image Width (pixels)")
    ax2.set_ylabel("Runtime (s)")
    ax2.set_title("Runtime vs. Image Width")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("graphs/increasing_data_subplots.png")
    plt.close()

    # ------------------------- Main -------------------------
if __name__ == "__main__":

    #plot_cuda_time_breakdown()
    #plot_mpi_comm_vs_comp()
    plot_runtime_comparison()
    plot_increasing_data()
    plot_combined_scaling()
    plot_weak_scaling_efficiencies()
    plot_openmp_weak_scaling_efficiencies()
    """plot_increasing_frames_runtime()
    plot_increasing_size_runtime()
    plot_mpi_speedup()
    plot_openmp_speedup()
    plot_combined_scaling()
    plot_weak_scaling_efficiency_frames()
    plot_weak_scaling_efficiency_size()
    plot_openmp_weak_scaling_efficiency_frames()
    plot_openmp_weak_scaling_efficiency_size()
    plot_cuda_time_breakdown()"""
    print("Graphs generated and saved in the 'graphs' directory.")
