import numpy as np
import pandas as pd
import time
import datetime
import random
import os
from rich.console import Console
from rich.table import Table
from rich.live import Live
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import uuid

class HealthDataGenerator:
    def __init__(self, 
                 anomaly_probability=0.03,
                 data_generation_interval=0.5,
                 window_size=100):
        self.console = Console()
        self.anomaly_probability = anomaly_probability
        self.data_generation_interval = data_generation_interval
        self.window_size = window_size
        
        # Base parameters for normal vital signs
        self.base_params = {
            'blood_pressure': {'mean': 120, 'std': 5},
            'heart_rate': {'mean': 75, 'std': 3},
            'resource_usage': {'mean': 98, 'std': 2}
        }
        
        # Define different types of anomalies
        self.anomaly_types = {
            'heart_attack': {
                'blood_pressure': {'shift': 40, 'std': 15},
                'heart_rate': {'shift': 50, 'std': 20},
                'resource_usage': {'shift': -30, 'std': 10},
                'duration': 10  # seconds
            },
            'hypertensive_crisis': {
                'blood_pressure': {'shift': 60, 'std': 10},
                'heart_rate': {'shift': 30, 'std': 15},
                'resource_usage': {'shift': -20, 'std': 5},
                'duration': 15
            },
            'hypotension': {
                'blood_pressure': {'shift': -30, 'std': 10},
                'heart_rate': {'shift': 20, 'std': 10},
                'resource_usage': {'shift': -15, 'std': 5},
                'duration': 12
            }
        }
        
        # Initialize data storage
        self.data = []
        self.current_anomaly = None
        self.anomaly_start_time = None
        
        # Setup output file
        self.output_dir = r"C:\Users\Mohamed\Desktop\proj\training_data"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, f"health_data_{uuid.uuid4()}.csv")
        
        # Initialize the CSV file with headers
        with open(self.output_file, 'w') as f:
            f.write("timestamp,blood_pressure,heart_rate,resource_usage,anomaly_type\n")
    
    def generate_reading(self):
        timestamp = datetime.datetime.now()
        
        # Check if we should start a new anomaly
        if self.current_anomaly is None and random.random() < self.anomaly_probability:
            self.current_anomaly = random.choice(list(self.anomaly_types.keys()))
            self.anomaly_start_time = time.time()
        
        # Generate readings based on current state
        readings = {}
        anomaly_type = self.current_anomaly if self.current_anomaly else "normal"
        
        for metric in self.base_params.keys():
            if self.current_anomaly:
                anomaly = self.anomaly_types[self.current_anomaly]
                shift = anomaly[metric]['shift']
                std = anomaly[metric]['std']
                base = self.base_params[metric]['mean'] + shift
                readings[metric] = max(0, np.random.normal(base, std))
            else:
                readings[metric] = max(0, np.random.normal(
                    self.base_params[metric]['mean'],
                    self.base_params[metric]['std']
                ))
        
        # Check if anomaly should end
        if self.current_anomaly:
            if time.time() - self.anomaly_start_time > self.anomaly_types[self.current_anomaly]['duration']:
                self.current_anomaly = None
                self.anomaly_start_time = None
        
        data_point = {
            'timestamp': timestamp,
            'blood_pressure': readings['blood_pressure'],
            'heart_rate': readings['heart_rate'],
            'resource_usage': readings['resource_usage'],
            'anomaly_type': anomaly_type
        }
        
        self.data.append(data_point)
        
        # Keep only recent data in memory
        if len(self.data) > self.window_size:
            self.data.pop(0)
        
        # Save to CSV
        with open(self.output_file, 'a') as f:
            f.write(f"{timestamp},{readings['blood_pressure']:.2f},{readings['heart_rate']:.2f},"
                   f"{readings['resource_usage']:.2f},{anomaly_type}\n")
        
        return data_point
    
    def create_rich_table(self, data_point):
        table = Table(title="Health Monitoring Data")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        status = "⚠️ ANOMALY: " + data_point['anomaly_type'] if data_point['anomaly_type'] != "normal" else "✅ Normal"
        status_style = "red" if data_point['anomaly_type'] != "normal" else "green"
        
        table.add_row("Timestamp", str(data_point['timestamp']), status)
        table.add_row("Blood Pressure", f"{data_point['blood_pressure']:.2f} mmHg", "")
        table.add_row("Heart Rate", f"{data_point['heart_rate']:.2f} bpm", "")
        table.add_row("Resource Usage", f"{data_point['resource_usage']:.2f}%", "")
        
        return table

class HealthDataVisualizer:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        
        # Setup the plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Real-time Health Monitoring')
        
        # Initialize empty lines
        self.lines = {
            'blood_pressure': self.ax1.plot([], [], 'r-', label='Blood Pressure')[0],
            'heart_rate': self.ax2.plot([], [], 'g-', label='Heart Rate')[0],
            'resource_usage': self.ax3.plot([], [], 'b-', label='Resource Usage')[0]
        }
        
        # Setup axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlim(0, data_generator.window_size)
            ax.grid(True)
            ax.legend()
        
        self.ax1.set_ylim(60, 200)
        self.ax2.set_ylim(40, 150)
        self.ax3.set_ylim(0, 120)
    
    def update(self, frame):
        data = self.data_generator.data
        if not data:
            return self.lines.values()
        
        x = range(len(data))
        for metric, line in self.lines.items():
            y = [d[metric] for d in data]
            line.set_data(x, y)
        
        return self.lines.values()

def main():
    # Initialize the data generator and visualizer
    generator = HealthDataGenerator()
    visualizer = HealthDataVisualizer(generator)
    
    # Setup the animation
    ani = FuncAnimation(visualizer.fig, visualizer.update, interval=50, blit=True)
    plt.show(block=False)
    
    # Main loop for data generation and console output
    with Live(generator.create_rich_table(generator.generate_reading()), refresh_per_second=4) as live:
        while True:
            data_point = generator.generate_reading()
            live.update(generator.create_rich_table(data_point))
            plt.pause(generator.data_generation_interval)

if __name__ == "__main__":
    main()
