import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('results.csv')

# Create a color mapping for each unique model base name (without bit specification)
model_bases = df['model_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1] if 'video' in x else x.split('_')[0])
unique_models = model_bases.unique()
colors = sns.color_palette("husl", len(unique_models))
color_map = dict(zip(unique_models, colors))

# Function to get color based on model name
def get_color(model_name):
    base = model_name.split('_')[0] + '_' + model_name.split('_')[1] if 'video' in model_name else model_name.split('_')[0]
    return color_map[base]

# Function to convert memory to GB
def bytes_to_gb(bytes_val):
    return bytes_val / (1024 ** 3)

# Get unique benchmarks
benchmarks = df['benchmark'].unique()

# Create a figure for each benchmark
plt.style.use('seaborn')
for i, benchmark in enumerate(benchmarks):
    benchmark_data = df[df['benchmark'] == benchmark]
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot bars on primary axis (accuracy)
    bars = ax1.bar(range(len(benchmark_data)), benchmark_data['accuracy'])
    
    # Set colors for each bar based on model name
    for bar, model_name in zip(bars, benchmark_data['model_name']):
        bar.set_color(get_color(model_name))
    
    # Customize primary axis (accuracy)
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Accuracy', color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    
    # Create secondary axis (memory utilization)
    ax2 = ax1.twinx()
    
    # Plot line on secondary axis (memory)
    memory_gb = benchmark_data['memory_utilization'].apply(bytes_to_gb)
    line = ax2.plot(range(len(benchmark_data)), memory_gb, 
                    color='darkred', marker='o', linewidth=2, 
                    linestyle='--', label='Memory Usage (GB)')
    
    # Customize secondary axis (memory)
    ax2.set_ylabel('Memory Usage (GB)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Set title and x-axis labels
    plt.title(f'{benchmark.upper()} Benchmark Results: Accuracy vs Memory Usage', pad=20)
    ax1.set_xticks(range(len(benchmark_data)))
    ax1.set_xticklabels(benchmark_data['model_name'], rotation=45, ha='right')
    
    # Add grid for better readability (only for primary axis)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    lines = ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    plt.savefig(f'benchmark_results/{benchmark}_results_with_memory.png', dpi=300, bbox_inches='tight')
    plt.close()


for i, benchmark in enumerate(benchmarks):
    benchmark_data = df[df['benchmark'] == benchmark]
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    bars = ax1.bar(range(len(benchmark_data)), benchmark_data['additional_results'])
    
    for bar, model_name in zip(bars, benchmark_data['model_name']):
        bar.set_color(get_color(model_name))
    
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Accuracy', color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    
    ax2 = ax1.twinx()
    
    memory_gb = benchmark_data['memory_utilization'].apply(bytes_to_gb)
    line = ax2.plot(range(len(benchmark_data)), memory_gb, 
                    color='darkred', marker='o', linewidth=2, 
                    linestyle='--', label='Memory Usage (GB)')
    
    ax2.set_ylabel('Memory Usage (GB)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    plt.title(f'{benchmark.upper()} Benchmark Results: Accuracy vs Memory Usage', pad=20)
    ax1.set_xticks(range(len(benchmark_data)))
    ax1.set_xticklabels(benchmark_data['model_name'], rotation=45, ha='right')
    
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    lines = ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    plt.savefig(f'benchmark_results/{benchmark}_results_with_bertF1.png', dpi=300, bbox_inches='tight')
    plt.close()


print("Plots have been generated and saved as PNG files.")