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

# Get unique benchmarks
benchmarks = df['benchmark'].unique()

# Create a figure for each benchmark
plt.style.use('seaborn')
for i, benchmark in enumerate(benchmarks):
    benchmark_data = df[df['benchmark'] == benchmark]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(benchmark_data)), benchmark_data['accuracy'])
    
    # Set colors for each bar based on model name
    for bar, model_name in zip(bars, benchmark_data['model_name']):
        bar.set_color(get_color(model_name))
    
    # Customize the plot
    plt.title(f'{benchmark.upper()} Benchmark Results', pad=20)
    plt.xlabel('Model Configuration')
    plt.ylabel('Accuracy')
    
    # Set x-axis labels (rotated for better readability)
    plt.xticks(range(len(benchmark_data)), 
               benchmark_data['model_name'],
               rotation=45,
               ha='right')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'benchmark_results/{benchmark}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Plots have been generated and saved as PNG files.")