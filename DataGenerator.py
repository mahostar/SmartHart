import numpy as np
import pandas as pd
import time
import datetime
import random
import os
import tkinter as tk
from tkinter import ttk
import uuid
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class HealthDataGenerator:
    def __init__(self):
        # Configurable parameters with realistic baseline values
        self.config = {
            'data_generation_interval': 0.5,  # seconds
            'window_size': 100,
            'anomaly_probability': 0.03
        }
        
        # More realistic vital signs parameters based on medical standards
        self.base_params = {
            'blood_pressure': {
                'systolic': {'mean': 120, 'std': 5, 'min': 70, 'max': 180},
                'diastolic': {'mean': 80, 'std': 3, 'min': 40, 'max': 120},
            },
            'heart_rate': {
                'mean': 75, 'std': 3, 'min': 40, 'max': 150,
                'respiratory_sinus_arrhythmia': 0.1  # Natural heart rate variation with breathing
            },
            'spo2': {
                'mean': 98, 'std': 0.5, 'min': 80, 'max': 100
            },
            'respiratory_rate': {
                'mean': 16, 'std': 1, 'min': 8, 'max': 30
            }
        }
        
        # Enhanced anomaly definitions with more realistic patterns
        self.anomaly_types = {
            'cardiac_arrest': {
                'blood_pressure': {'systolic_shift': -40, 'diastolic_shift': -30, 'std': 15},
                'heart_rate': {'shift': -30, 'std': 20},
                'spo2': {'shift': -15, 'std': 5},
                'respiratory_rate': {'shift': 8, 'std': 3},
                'duration': 15,
                'onset_speed': 'rapid'
            },
            'hypertensive_crisis': {
                'blood_pressure': {'systolic_shift': 60, 'diastolic_shift': 40, 'std': 10},
                'heart_rate': {'shift': 30, 'std': 15},
                'spo2': {'shift': -5, 'std': 2},
                'respiratory_rate': {'shift': 6, 'std': 2},
                'duration': 20,
                'onset_speed': 'gradual'
            },
            'septic_shock': {
                'blood_pressure': {'systolic_shift': -30, 'diastolic_shift': -20, 'std': 10},
                'heart_rate': {'shift': 40, 'std': 10},
                'spo2': {'shift': -10, 'std': 3},
                'respiratory_rate': {'shift': 10, 'std': 3},
                'duration': 25,
                'onset_speed': 'gradual'
            }
        }
        
        self.data = []
        self.current_anomaly = None
        self.anomaly_start_time = None
        self.last_update = time.time()
        
        # Setup output directory
        self.output_dir = "health_monitoring_data"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, f"health_data_{uuid.uuid4()}.csv")
        
        # Initialize CSV file
        self.initialize_csv()

    def initialize_csv(self):
        headers = ["timestamp", "systolic_bp", "diastolic_bp", "heart_rate", 
                  "spo2", "respiratory_rate", "anomaly_type"]
        with open(self.output_file, 'w') as f:
            f.write(','.join(headers) + '\n')

    def apply_physiological_constraints(self, readings):
        """Apply realistic physiological relationships between vital signs"""
        # Heart rate affects blood pressure
        if readings['heart_rate'] > self.base_params['heart_rate']['mean']:
            factor = (readings['heart_rate'] - self.base_params['heart_rate']['mean']) / 50
            readings['blood_pressure']['systolic'] *= (1 + 0.1 * factor)
            readings['blood_pressure']['diastolic'] *= (1 + 0.05 * factor)
        
        # SpO2 affects heart rate
        if readings['spo2'] < 90:
            readings['heart_rate'] *= (1 + (90 - readings['spo2']) / 100)
        
        return readings

    def generate_reading(self):
        current_time = time.time()
        if current_time - self.last_update < self.config['data_generation_interval']:
            return None
        
        self.last_update = current_time
        timestamp = datetime.datetime.now()
        
        # Initialize readings
        readings = {
            'blood_pressure': {
                'systolic': 0,
                'diastolic': 0
            },
            'heart_rate': 0,
            'spo2': 0,
            'respiratory_rate': 0
        }
        
        # Check for new anomaly
        if self.current_anomaly is None and random.random() < self.config['anomaly_probability']:
            self.current_anomaly = random.choice(list(self.anomaly_types.keys()))
            self.anomaly_start_time = current_time
        
        # Generate readings
        if self.current_anomaly:
            anomaly = self.anomaly_types[self.current_anomaly]
            progress = (current_time - self.anomaly_start_time) / anomaly['duration']
            
            # Apply anomaly effects with onset speed
            for vital in readings.keys():
                if vital == 'blood_pressure':
                    sys_shift = anomaly['blood_pressure']['systolic_shift']
                    dia_shift = anomaly['blood_pressure']['diastolic_shift']
                    std = anomaly['blood_pressure']['std']
                    
                    readings[vital]['systolic'] = np.random.normal(
                        self.base_params[vital]['systolic']['mean'] + sys_shift * progress,
                        std
                    )
                    readings[vital]['diastolic'] = np.random.normal(
                        self.base_params[vital]['diastolic']['mean'] + dia_shift * progress,
                        std
                    )
                else:
                    shift = anomaly[vital]['shift'] if vital in anomaly else 0
                    std = anomaly[vital]['std'] if vital in anomaly else self.base_params[vital]['std']
                    readings[vital] = np.random.normal(
                        self.base_params[vital]['mean'] + shift * progress,
                        std
                    )
            
            # Check if anomaly should end
            if current_time - self.anomaly_start_time > anomaly['duration']:
                self.current_anomaly = None
                self.anomaly_start_time = None
        else:
            # Generate normal readings
            for vital in readings.keys():
                if vital == 'blood_pressure':
                    readings[vital]['systolic'] = np.random.normal(
                        self.base_params[vital]['systolic']['mean'],
                        self.base_params[vital]['systolic']['std']
                    )
                    readings[vital]['diastolic'] = np.random.normal(
                        self.base_params[vital]['diastolic']['mean'],
                        self.base_params[vital]['diastolic']['std']
                    )
                else:
                    readings[vital] = np.random.normal(
                        self.base_params[vital]['mean'],
                        self.base_params[vital]['std']
                    )
        
        # Apply physiological constraints
        readings = self.apply_physiological_constraints(readings)
        
        # Ensure values are within realistic ranges
        readings['blood_pressure']['systolic'] = np.clip(
            readings['blood_pressure']['systolic'],
            self.base_params['blood_pressure']['systolic']['min'],
            self.base_params['blood_pressure']['systolic']['max']
        )
        readings['blood_pressure']['diastolic'] = np.clip(
            readings['blood_pressure']['diastolic'],
            self.base_params['blood_pressure']['diastolic']['min'],
            self.base_params['blood_pressure']['diastolic']['max']
        )
        
        for vital in ['heart_rate', 'spo2', 'respiratory_rate']:
            readings[vital] = np.clip(
                readings[vital],
                self.base_params[vital]['min'],
                self.base_params[vital]['max']
            )
        
        # Create data point
        data_point = {
            'timestamp': timestamp,
            'systolic_bp': readings['blood_pressure']['systolic'],
            'diastolic_bp': readings['blood_pressure']['diastolic'],
            'heart_rate': readings['heart_rate'],
            'spo2': readings['spo2'],
            'respiratory_rate': readings['respiratory_rate'],
            'anomaly_type': self.current_anomaly if self.current_anomaly else "normal"
        }
        
        self.data.append(data_point)
        
        # Maintain window size
        if len(self.data) > self.config['window_size']:
            self.data.pop(0)
        
        # Save to CSV
        self.save_to_csv(data_point)
        
        return data_point

    def save_to_csv(self, data_point):
        with open(self.output_file, 'a') as f:
            f.write(f"{data_point['timestamp']},{data_point['systolic_bp']:.2f},"
                   f"{data_point['diastolic_bp']:.2f},{data_point['heart_rate']:.2f},"
                   f"{data_point['spo2']:.2f},{data_point['respiratory_rate']:.2f},"
                   f"{data_point['anomaly_type']}\n")

class HealthMonitorDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Health Monitoring Dashboard")
        self.root.state('zoomed')
        
        # Add this line to handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.generator = HealthDataGenerator()
        self.setup_ui()
        self.running = True
        
        self.data_thread = threading.Thread(target=self.update_data, daemon=True)
        self.data_thread.start()

    def setup_ui(self):
        # Create main containers
        control_frame = ttk.Frame(self.root, padding="5")
        graph_frame = ttk.Frame(self.root, padding="5")
        
        # Important: Setup graphs BEFORE controls
        self.setup_graphs(graph_frame)
        self.setup_controls(control_frame)
        
        # Pack frames after both are set up
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def setup_controls(self, parent):
        # Title
        ttk.Label(parent, text="Control Panel", font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        # Data Generation Speed
        ttk.Label(parent, text="Data Generation Interval (seconds)").pack(pady=5)
        speed_scale = ttk.Scale(parent, from_=0.1, to=2.0, 
                              command=lambda v: self.update_config('data_generation_interval', float(v)))
        speed_scale.set(self.generator.config['data_generation_interval'])
        speed_scale.pack(fill=tk.X, padx=5)
        
        # Window Size
        ttk.Label(parent, text="Window Size (samples)").pack(pady=5)
        window_scale = ttk.Scale(parent, from_=50, to=500,
                               command=lambda v: self.update_config('window_size', int(float(v))))
        window_scale.set(self.generator.config['window_size'])
        window_scale.pack(fill=tk.X, padx=5)
        
        # Anomaly Probability
        ttk.Label(parent, text="Anomaly Probability").pack(pady=5)
        prob_scale = ttk.Scale(parent, from_=0, to=0.2,
                             command=lambda v: self.update_config('anomaly_probability', float(v)))
        prob_scale.set(self.generator.config['anomaly_probability'])
        prob_scale.pack(fill=tk.X, padx=5)
        
        # Current Status Frame
        status_frame = ttk.LabelFrame(parent, text="Current Status", padding="5")
        status_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.status_labels = {
            'anomaly': ttk.Label(status_frame, text="Status: Normal"),
            'bp': ttk.Label(status_frame, text="BP: --/-- mmHg"),
            'hr': ttk.Label(status_frame, text="HR: -- bpm"),
            'spo2': ttk.Label(status_frame, text="SpO2: --%"),
            'rr': ttk.Label(status_frame, text="RR: -- /min")
        }
        
        for label in self.status_labels.values():
            label.pack(pady=2)

    def setup_graphs(self, parent):
        # Create figure with white background
        self.fig = Figure(figsize=(12, 8), facecolor='white')
        self.fig.subplots_adjust(hspace=0.3)
        
        # Create subplots
        self.axes = {
            'bp': self.fig.add_subplot(411),
            'hr': self.fig.add_subplot(412),
            'spo2': self.fig.add_subplot(413),
            'rr': self.fig.add_subplot(414)
        }
        
        # Configure axes
        for ax in self.axes.values():
            ax.set_facecolor('white')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(labelcolor='black')
        
        # Set specific configurations for each plot
        self.axes['bp'].set_ylabel('Blood Pressure\n(mmHg)', color='black')
        self.axes['hr'].set_ylabel('Heart Rate\n(bpm)', color='black')
        self.axes['spo2'].set_ylabel('SpO2\n(%)', color='black')
        self.axes['rr'].set_ylabel('Respiratory Rate\n(/min)', color='black')
        
        # Initialize empty lines
        self.lines = {
            'systolic': self.axes['bp'].plot([], [], 'r-', label='Systolic')[0],
            'diastolic': self.axes['bp'].plot([], [], 'b-', label='Diastolic')[0],
            'hr': self.axes['hr'].plot([], [], 'g-', label='Heart Rate')[0],
            'spo2': self.axes['spo2'].plot([], [], 'm-', label='SpO2')[0],
            'rr': self.axes['rr'].plot([], [], 'c-', label='Respiratory Rate')[0]
        }

        # Set up legends
        for ax in self.axes.values():
            ax.legend(loc='upper right', facecolor='white', edgecolor='black')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize animation
        self.ani = FuncAnimation(
            self.fig, 
            self.update_plots, 
            interval=50, 
            blit=True,
            save_count=100  # Add this parameter
        )

    def update_config(self, param, value):
        """Update configuration parameters in real-time"""
        self.generator.config[param] = value
        if param == 'window_size':
            # Update x-axis limits for all plots
            for ax in self.axes.values():
                ax.set_xlim(0, value)
            self.canvas.draw()

    def update_plots(self, frame):
        """Update all plots with new data"""
        data = self.generator.data
        if not data:
            return self.lines.values()
        
        x = range(len(data))
        
        # Update blood pressure lines
        systolic_data = [d['systolic_bp'] for d in data]
        diastolic_data = [d['diastolic_bp'] for d in data]
        self.lines['systolic'].set_data(x, systolic_data)
        self.lines['diastolic'].set_data(x, diastolic_data)
        
        # Update other vital signs
        self.lines['hr'].set_data(x, [d['heart_rate'] for d in data])
        self.lines['spo2'].set_data(x, [d['spo2'] for d in data])
        self.lines['rr'].set_data(x, [d['respiratory_rate'] for d in data])
        
        # Set y-axis limits dynamically
        if data:
            self.axes['bp'].set_ylim(min(min(diastolic_data) - 10, 40),
                                   max(max(systolic_data) + 10, 200))
            self.axes['hr'].set_ylim(min(min(d['heart_rate'] for d in data) - 5, 40),
                                   max(max(d['heart_rate'] for d in data) + 5, 150))
            self.axes['spo2'].set_ylim(min(min(d['spo2'] for d in data) - 2, 80),
                                     max(max(d['spo2'] for d in data) + 2, 100))
            self.axes['rr'].set_ylim(min(min(d['respiratory_rate'] for d in data) - 2, 8),
                                   max(max(d['respiratory_rate'] for d in data) + 2, 30))
        
        return self.lines.values()

    def update_status_labels(self, data_point):
        """Update the status labels with current readings"""
        if data_point['anomaly_type'] == "normal":
            self.status_labels['anomaly'].config(text="Status: Normal", foreground="green")
        else:
            self.status_labels['anomaly'].config(
                text=f"Status: ⚠️ {data_point['anomaly_type'].replace('_', ' ').title()}",
                foreground="red"
            )
        
        self.status_labels['bp'].config(
            text=f"BP: {data_point['systolic_bp']:.0f}/{data_point['diastolic_bp']:.0f} mmHg"
        )
        self.status_labels['hr'].config(
            text=f"HR: {data_point['heart_rate']:.0f} bpm"
        )
        self.status_labels['spo2'].config(
            text=f"SpO2: {data_point['spo2']:.0f}%"
        )
        self.status_labels['rr'].config(
            text=f"RR: {data_point['respiratory_rate']:.0f} /min"
        )

    def update_data(self):
        """Update data in a separate thread"""
        while self.running:
            data_point = self.generator.generate_reading()
            if data_point:
                # Update status labels in the main thread
                self.root.after(0, self.update_status_labels, data_point)
            time.sleep(0.05)  # Small sleep to prevent excessive CPU usage
    
    def on_closing(self):
        """Handle window closing event"""
        self.running = False
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the dashboard"""
        try:
            self.root.mainloop()
        finally:
            self.running = False

def main():
    dashboard = HealthMonitorDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()