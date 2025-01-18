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
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
warnings.filterwarnings("ignore", category=UserWarning)

class HealthPredictorDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Health Monitoring Dashboard with ML Prediction")
        self.root.state('zoomed')
        
        # Load ML model and preprocessing objects
        self.model = load_model("C:/Users/Mohamed/Desktop/proj/model/health_monitor_model.h5")
        # Fix: Add allow_pickle=True parameter
        self.label_encoder_classes = np.load(
            "C:/Users/Mohamed/Desktop/proj/model/label_encoder_classes.npy",
            allow_pickle=True
        )
        self.scaler = joblib.load("C:/Users/Mohamed/Desktop/proj/model/scaler.pkl")
        
        # Initialize data generator and prediction tracking
        self.generator = HealthDataGenerator()
        self.prediction_buffer = []
        self.accuracy_history = []
        self.running = True
        
        # Setup UI
        self.setup_ui()
        
        # Start data generation thread
        self.data_thread = threading.Thread(target=self.update_data, daemon=True)
        self.data_thread.start()
        
        # Add window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Create main containers
        control_frame = ttk.Frame(self.root, padding="5")
        graph_frame = ttk.Frame(self.root, padding="5")
        prediction_frame = ttk.Frame(self.root, padding="5")
        
        # Setup components
        self.setup_graphs(graph_frame)
        self.setup_controls(control_frame)
        self.setup_prediction_panel(prediction_frame)
        
        # Pack frames
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        prediction_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

    def setup_prediction_panel(self, parent):
        # Prediction Status Frame
        pred_status = ttk.LabelFrame(parent, text="Prediction Status", padding="10")
        pred_status.pack(fill=tk.X, padx=5, pady=5)
        
        # True vs Predicted Labels
        self.true_label = ttk.Label(pred_status, text="True Condition: Normal", font=('Helvetica', 12))
        self.true_label.pack(pady=5)
        
        self.pred_label = ttk.Label(pred_status, text="Predicted: Normal", font=('Helvetica', 12))
        self.pred_label.pack(pady=5)
        
        # Accuracy Frame
        accuracy_frame = ttk.LabelFrame(parent, text="Model Performance", padding="10")
        accuracy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.current_accuracy = ttk.Label(accuracy_frame, 
                                        text="Current Accuracy: 100%",
                                        font=('Helvetica', 12))
        self.current_accuracy.pack(pady=5)
        
        # Add accuracy plot
        self.acc_fig = Figure(figsize=(6, 4), facecolor='white')
        self.acc_ax = self.acc_fig.add_subplot(111)
        self.acc_ax.set_ylim(0, 100)
        self.acc_ax.set_title("Accuracy Over Time")
        self.acc_ax.set_ylabel("Accuracy (%)")
        self.acc_ax.grid(True)
        
        self.acc_canvas = FigureCanvasTkAgg(self.acc_fig, master=accuracy_frame)
        self.acc_canvas.draw()
        self.acc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_controls(self, parent):
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
            ax.legend(loc='upper right')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize animation
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=50, blit=True)

    def make_prediction(self, data_point):
        # Add the new data point to the prediction buffer
        self.prediction_buffer.append([
            data_point['systolic_bp'],
            data_point['diastolic_bp'],
            data_point['heart_rate'],
            data_point['spo2'],
            data_point['respiratory_rate'],
            data_point['systolic_bp'] - data_point['diastolic_bp'],  # pulse_pressure
            data_point['heart_rate'] / (data_point['systolic_bp'] + 1e-6)  # shock_index
        ])
        
        # Keep only the last 10 readings
        if len(self.prediction_buffer) > 10:
            self.prediction_buffer.pop(0)
        
        # Only predict when we have enough data
        if len(self.prediction_buffer) == 10:
            # Scale the data
            scaled_data = self.scaler.transform(self.prediction_buffer)
            # Reshape for model input (1, 10, 7)
            model_input = scaled_data.reshape(1, 10, -1)
            # Make prediction
            pred_proba = self.model.predict(model_input, verbose=0)
            pred_class = np.argmax(pred_proba)
            predicted_label = self.label_encoder_classes[pred_class]
            
            # Update accuracy tracking
            correct = predicted_label == data_point['anomaly_type']
            self.accuracy_history.append(correct)
            if len(self.accuracy_history) > 100:  # Keep last 100 predictions
                self.accuracy_history.pop(0)
            
            current_accuracy = (sum(self.accuracy_history) / len(self.accuracy_history)) * 100
            
            # Update UI with prediction results
            self.root.after(0, self.update_prediction_display,
                          data_point['anomaly_type'],
                          predicted_label,
                          current_accuracy)

    def update_prediction_display(self, true_label, pred_label, accuracy):
        # Maximum number of points to display on the graph
        MAX_DISPLAY_POINTS = 50
        
        # Update labels
        self.true_label.config(
            text=f"True Condition: {true_label}",
            foreground="black"
        )
        self.pred_label.config(
            text=f"Predicted: {pred_label}",
            foreground="green" if true_label == pred_label else "red"
        )
        
        # Update accuracy display with actual current accuracy
        self.current_accuracy.config(text=f"Current Accuracy: {accuracy:.1f}%")
        
        # Update accuracy plot with raw data
        self.acc_ax.clear()
        
        # Get the most recent data points
        display_history = self.accuracy_history[-MAX_DISPLAY_POINTS:]
        x_values = range(len(display_history))
        y_values = [1 if val else 0 for val in display_history]
        # Convert to percentage
        y_values = [y * 100 for y in y_values]
        
        # Plot line only
        self.acc_ax.plot(x_values, y_values, 'b-', linewidth=2)
        
        # Fixed y-axis from 0 to 100 percent
        self.acc_ax.set_ylim(-5, 105)
        
        # Add horizontal lines at 0% and 100%
        self.acc_ax.axhline(y=0, color='red', linestyle='-', alpha=0.2)
        self.acc_ax.axhline(y=100, color='green', linestyle='-', alpha=0.2)
        
        # Customize appearance
        self.acc_ax.set_title("Prediction Accuracy (Real-time)", fontsize=12)
        self.acc_ax.set_ylabel("Accuracy (%)", fontsize=10)
        self.acc_ax.set_xlabel("Recent Predictions", fontsize=10)
        self.acc_ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add current running accuracy as text
        avg_accuracy = sum(display_history) / len(display_history) * 100
        self.acc_ax.text(0.02, 0.98, f'Running Accuracy: {avg_accuracy:.1f}%', 
                        transform=self.acc_ax.transAxes,
                        verticalalignment='top',
                        fontsize=10)
        
        # Redraw canvas
        self.acc_canvas.draw()

    def update_config(self, param, value):
        self.generator.config[param] = value

    def update_plots(self, frame):
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
        
        # Set x-axis limits
        for ax in self.axes.values():
            ax.set_xlim(0, len(data))
        
        return self.lines.values()

    def update_data(self):
        while self.running:
            data_point = self.generator.generate_reading()
            if data_point:
                # Update status labels
                self.root.after(0, self.update_status_labels, data_point)
                # Make prediction
                self.make_prediction(data_point)
            time.sleep(0.05)

    def update_status_labels(self, data_point):
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

    def on_closing(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.running = False


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
                'respiratory_sinus_arrhythmia': 0.1
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
        
        return data_point


def main():
    dashboard = HealthPredictorDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()