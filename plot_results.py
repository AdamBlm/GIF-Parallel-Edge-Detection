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

# ------------------------- Graph: Weak Scaling Efficiency (Increasing Frames) -------------------------
def plot_weak_scaling_efficiency_frames():
    """
    Reads 'data/weak_scaling_frames.csv' which has columns:
    NumProcs, NumFrames, TotalTime.
    Computes weak scaling efficiency as: Efficiency = T(1) / T(N)
    and plots efficiency vs. number of processes.
    """
    try:
        df = pd.read_csv("data/weak_scaling_frames.csv")
    except Exception as e:
        print("data/weak_scaling_frames.csv not found.")
        return
    
    # Convert columns to numeric
    df["NumProcs"] = pd.to_numeric(df["NumProcs"], errors="coerce")
    df["TotalTime"] = pd.to_numeric(df["TotalTime"], errors="coerce")
    
    # Use the TotalTime for NP=1 as baseline
    baseline_time = df[df["NumProcs"] == 1]["TotalTime"].mean()
    df["Efficiency"] = baseline_time / df["TotalTime"]
    
    plt.figure(figsize=(8,6))
    plt.plot(df["NumProcs"].to_numpy(), df["Efficiency"].to_numpy(), marker='o')
    plt.xlabel("Number of Processes")
    plt.ylabel("Weak Scaling Efficiency")
    plt.title("Weak Scaling Efficiency (Increasing Frames)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/weak_scaling_efficiency_frames.png")
    plt.close()

# ------------------------- Graph: Weak Scaling Efficiency (Increasing Resolution) -------------------------
def plot_weak_scaling_efficiency_size():
    """
    Reads 'data/weak_scaling_size.csv' which has columns:
    NumProcs, Resolution, TotalTime.
    Computes weak scaling efficiency as: Efficiency = T(1) / T(N)
    and plots efficiency vs. number of processes.
    """
    try:
        df = pd.read_csv("data/weak_scaling_size.csv")
    except Exception as e:
        print("data/weak_scaling_size.csv not found.")
        return
    
    df["NumProcs"] = pd.to_numeric(df["NumProcs"], errors="coerce")
    df["TotalTime"] = pd.to_numeric(df["TotalTime"], errors="coerce")
    
    baseline_time = df[df["NumProcs"] == 1]["TotalTime"].mean()
    df["Efficiency"] = baseline_time / df["TotalTime"]
    
    plt.figure(figsize=(8,6))
    plt.plot(df["NumProcs"].to_numpy(), df["Efficiency"].to_numpy(), marker='o')
    plt.xlabel("Number of Processes")
    plt.ylabel("Weak Scaling Efficiency")
    plt.title("Weak Scaling Efficiency (Increasing Resolution)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/weak_scaling_efficiency_size.png")
    plt.close()



# ------------------------- Graph: OpenMP Weak Scaling Efficiency (Increasing Frames) -------------------------
def plot_openmp_weak_scaling_efficiency_frames():
    """
    Reads 'data/openmp_weak_scaling_frames.csv' with columns:
    Threads, NumFrames, TotalTime.
    Computes weak scaling efficiency as Efficiency = T(1) / T(n) and plots it versus thread count.
    """
    try:
        df = pd.read_csv("data/openmp_weak_scaling_frames.csv")
    except Exception as e:
        print("data/openmp_weak_scaling_frames.csv not found.")
        return

    # Convert columns to numeric
    df["Threads"] = pd.to_numeric(df["Threads"], errors="coerce")
    df["TotalTime"] = pd.to_numeric(df["TotalTime"], errors="coerce")
    
    # Use the runtime at 1 thread as the baseline
    baseline_time = df[df["Threads"] == 1]["TotalTime"].mean()
    df["Efficiency"] = baseline_time / df["TotalTime"]

    plt.figure(figsize=(8,6))
    plt.plot(df["Threads"].to_numpy(), df["Efficiency"].to_numpy(), marker='o', color="#1f77b4")
    plt.xlabel("Number of Threads")
    plt.ylabel("Weak Scaling Efficiency")
    plt.title("OpenMP Weak Scaling Efficiency (Increasing Frames)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/openmp_weak_scaling_efficiency_frames.png")
    plt.close()

# ------------------------- Graph: OpenMP Weak Scaling Efficiency (Increasing Resolution) -------------------------
def plot_openmp_weak_scaling_efficiency_size():
    """
    Reads 'data/openmp_weak_scaling_size.csv' with columns:
    Threads, Resolution, TotalTime.
    Computes weak scaling efficiency as Efficiency = T(1) / T(n) and plots it versus thread count.
    """
    try:
        df = pd.read_csv("data/openmp_weak_scaling_size.csv")
    except Exception as e:
        print("data/openmp_weak_scaling_size.csv not found.")
        return

    df["Threads"] = pd.to_numeric(df["Threads"], errors="coerce")
    df["TotalTime"] = pd.to_numeric(df["TotalTime"], errors="coerce")
    
    baseline_time = df[df["Threads"] == 1]["TotalTime"].mean()
    df["Efficiency"] = baseline_time / df["TotalTime"]

    plt.figure(figsize=(8,6))
    plt.plot(df["Threads"].to_numpy(), df["Efficiency"].to_numpy(), marker='o', color="#ff7f0e")
    plt.xlabel("Number of Threads")
    plt.ylabel("Weak Scaling Efficiency")
    plt.title("OpenMP Weak Scaling Efficiency (Increasing Resolution)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/openmp_weak_scaling_efficiency_size.png")
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
    This function reads the CSV file 'data/compare_all.csv' which contains
    the runtime results for each GIF image and each implementation:
    Image,Sequential,OpenMP,MPI+OpenMP,Hybrid,CUDA,MPI Frames,MPI Domain

    It then creates a grouped bar chart with one group per image.
    """
    try:
        df = pd.read_csv("data/compare_all.csv")
    except Exception as e:
        print("compare_all.csv not found in data folder.")
        return
    
    # Convert runtime columns to numeric (non-numeric entries become NaN)
    versions = ["Sequential", "OpenMP", "MPI+OpenMP", "Hybrid", "CUDA", "MPI Frames", "MPI Domain"]
    for ver in versions:
        df[ver] = pd.to_numeric(df[ver], errors="coerce")
    
    images = df["Image"]
    n = len(images)
    ind = np.arange(n)  # x locations for the groups
    width = 0.12        # width of each bar (adjusted for 7 groups)

    fig, ax = plt.subplots(figsize=(14,6))
    
    # Calculate offsets for the grouped bars
    offsets = np.arange(len(versions)) * width - (len(versions)-1)*width/2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    
    for i, ver in enumerate(versions):
        ax.bar(ind + offsets[i], df[ver], width, label=ver, color=colors[i])
    
    ax.set_xlabel("GIF Image Name")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Comparison per GIF across Implementations")
    ax.set_xticks(ind)
    ax.set_xticklabels(images, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("graphs/runtime_comparison.png")
    plt.close()

# ------------------------- Graph 7: Runtime vs. Number of Frames -------------------------
def plot_increasing_frames_runtime():
    """
    This function reads the CSV file 'data/increasing_frames.csv' which contains
    the runtime results for GIFs with increasing frame counts.
    The CSV should have columns: Image, NumFrames, OpenMP, MPI_Domain, CUDA.
    It then plots a line graph with NumFrames on the x-axis and runtime (s) on the y-axis.
    """
    try:
        df = pd.read_csv("data/increasing_frames.csv")
    except Exception as e:
        print("data/increasing_frames.csv not found.")
        return

    # Convert NumFrames to numeric
    df["NumFrames"] = pd.to_numeric(df["NumFrames"], errors="coerce")

    # Convert series to numpy arrays to avoid multi-dimensional indexing error
    num_frames = df["NumFrames"].to_numpy()
    openmp_time = df["OpenMP"].to_numpy()
    mpi_domain_time = df["MPI_Domain"].to_numpy()
    cuda_time = df["CUDA"].to_numpy()

    plt.figure(figsize=(10,6))
    plt.plot(num_frames, openmp_time, marker='o', label="OpenMP")
    plt.plot(num_frames, mpi_domain_time, marker='o', label="MPI Domain")
    plt.plot(num_frames, cuda_time, marker='o', label="CUDA")
    plt.xlabel("Number of Frames")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Number of Frames\n(OpenMP vs MPI Domain vs CUDA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/increasing_frames_runtime.png")
    plt.close()

# ------------------------- Graph 8: Runtime vs. Image Resolution -------------------------
def plot_increasing_size_runtime():
    """
    This function reads the CSV file 'data/increasing_size.csv' which contains
    the runtime results for GIFs with increasing resolution.
    The CSV should have columns: Image, Resolution, OpenMP, MPI_Domain, CUDA.
    The 'Resolution' is assumed to be in the format 'WidthxHeight' (e.g., "100x100").
    The function extracts the width and plots runtime (s) versus image width.
    """
    try:
        df = pd.read_csv("data/increasing_size.csv")
    except Exception as e:
        print("data/increasing_size.csv not found.")
        return

    # Extract the width from the Resolution column (assumes format "WidthxHeight")
    try:
        df["Width"] = df["Resolution"].str.split("x", expand=True)[0].astype(float)
    except Exception as e:
        print("Error processing the Resolution column:", e)
        return

    # Convert series to numpy arrays
    width = df["Width"].to_numpy()
    openmp_time = df["OpenMP"].to_numpy()
    mpi_domain_time = df["MPI_Domain"].to_numpy()
    cuda_time = df["CUDA"].to_numpy()

    plt.figure(figsize=(10,6))
    plt.plot(width, openmp_time, marker='o', label="OpenMP")
    plt.plot(width, mpi_domain_time, marker='o', label="MPI Domain")
    plt.plot(width, cuda_time, marker='o', label="CUDA")
    plt.xlabel("Image Width (pixels)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Image Resolution\n(OpenMP vs MPI Domain vs CUDA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/increasing_size_runtime.png")
    plt.close()

    # ------------------------- Main -------------------------
if __name__ == "__main__":

    #plot_cuda_time_breakdown()
    #plot_mpi_comm_vs_comp()
    plot_runtime_comparison()
    plot_increasing_frames_runtime()
    plot_increasing_size_runtime()
    plot_mpi_speedup()
    plot_openmp_speedup()
    plot_combined_scaling()
    plot_weak_scaling_efficiency_frames()
    plot_weak_scaling_efficiency_size()
    plot_openmp_weak_scaling_efficiency_frames()
    plot_openmp_weak_scaling_efficiency_size()
    plot_cuda_time_breakdown()
    print("Graphs generated and saved in the 'graphs' directory.")
