#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------- Graph 1: Run Time vs. Image Size -------------------------
def plot_runtime_vs_imagesize():
    # For this example, we assume that image names encode the size,
    # e.g. "320x200_..." or you have an auxiliary CSV with sizes.
    # Here we use the sequential CSV as an example.
    seq_df = pd.read_csv("data/sequential_baseline.csv")
    
    # Use the DataFrame index as a proxy for image size
    seq_df['ImageIndex'] = seq_df.index
    
    plt.figure(figsize=(10,6))
    plt.plot(seq_df['ImageIndex'].values, seq_df['TotalTime'].values, label='Sequential', marker='o')
    
    cuda_df = pd.read_csv("data/cuda_results.csv")
    cuda_df['ImageIndex'] = cuda_df.index
    plt.plot(cuda_df['ImageIndex'].values, cuda_df['TotalTime'].values, label='CUDA', marker='o')
    
    mpi_frame_df = pd.read_csv("data/mpi_frame_distribution.csv")
    # Average over NumProcs if needed:
    mpi_frame_avg = mpi_frame_df.groupby("ImageName")["TotalTime"].mean().reset_index()
    mpi_frame_avg['ImageIndex'] = np.arange(len(mpi_frame_avg))
    plt.plot(mpi_frame_avg['ImageIndex'].values, mpi_frame_avg['TotalTime'].values, label='MPI Frame', marker='o')
    
    mpi_domain_df = pd.read_csv("data/domain_decomposition_results.csv")
    mpi_domain_avg = mpi_domain_df.groupby("ImageName")["TotalTime"].mean().reset_index()
    mpi_domain_avg['ImageIndex'] = np.arange(len(mpi_domain_avg))
    plt.plot(mpi_domain_avg['ImageIndex'].values, mpi_domain_avg['TotalTime'].values, label='MPI Domain', marker='o')
    
    plt.xlabel("Image Index (proxy for image size)")
    plt.ylabel("Total Run Time (s)")
    plt.title("Run Time vs. Image Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/runtime_vs_imagesize.png")
    plt.close()

# ------------------------- Graph 2: Speedup vs. Number of Processes -------------------------
def plot_speedup_strong_scaling():
    seq_df = pd.read_csv("data/sequential_baseline.csv")
    # For a chosen image (say the first one), get its total time.
    base_time = seq_df.loc[0, "TotalTime"]
    
    mpi_frame_df = pd.read_csv("data/mpi_frame_distribution.csv")
    mpi_domain_df = pd.read_csv("data/domain_decomposition_results.csv")
    
    # For each NP value, compute average total time and then speedup.
    frame_speedup = mpi_frame_df.groupby("NumProcs")["TotalTime"].mean().apply(lambda t: base_time / t)
    domain_speedup = mpi_domain_df.groupby("NumProcs")["TotalTime"].mean().apply(lambda t: base_time / t)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(frame_speedup.index.values, frame_speedup.values, label="MPI Frame Distribution", marker='o')
    ax.plot(domain_speedup.index.values, domain_speedup.values, label="MPI Domain Decomposition", marker='o')
    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Speedup (Sequential Time / Parallel Time)")
    ax.set_title("Strong Scaling: Speedup vs. Number of Processes")
    ax.legend()
    ax.grid(True)
    plt.savefig("graphs/speedup_vs_procs.png")
    plt.close()

# ------------------------- Graph 3: Weak Scaling Efficiency -------------------------
def plot_weak_scaling():
    # Here we simulate weak scaling using one of the MPI CSVs.
    mpi_frame_df = pd.read_csv("data/mpi_frame_distribution.csv")
    # For weak scaling, efficiency = T(1)/T(N) when workload per process is constant.
    baseline_time = mpi_frame_df[mpi_frame_df["NumProcs"] == 1]["TotalTime"].mean()
    mpi_frame_df["Efficiency"] = mpi_frame_df["TotalTime"].apply(lambda t: baseline_time / t)
    
    # Average efficiency per NP
    efficiency = mpi_frame_df.groupby("NumProcs")["Efficiency"].mean()
    
    plt.figure(figsize=(8,6))
    plt.plot(efficiency.index.values, efficiency.values, marker='o')
    plt.xlabel("Number of Processes")
    plt.ylabel("Weak Scaling Efficiency")
    plt.title("Weak Scaling Efficiency")
    plt.grid(True)
    plt.savefig("graphs/weak_scaling_efficiency.png")
    plt.close()


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

# ------------------------- Main -------------------------
if __name__ == "__main__":
    plot_runtime_vs_imagesize()
    plot_speedup_strong_scaling()
    plot_weak_scaling()
    plot_cuda_time_breakdown()
    plot_mpi_comm_vs_comp()
    print("Graphs generated and saved in the 'graphs' directory.")
