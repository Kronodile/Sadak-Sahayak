import os
import csv
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ADASLogger:
    """
    Logger for ADAS system - records events, metrics and can generate reports
    """
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        # Create directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_log_path = os.path.join(log_dir, f"events_{self.timestamp}.csv")
        self.metrics_log_path = os.path.join(log_dir, f"metrics_{self.timestamp}.csv")
        
        # Initialize event log
        with open(self.event_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'event_type', 'details', 'severity'])
        
        # Initialize metrics log
        with open(self.metrics_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'metric_name', 'value'])
        
        # Initialize event counters
        self.event_counts = {
            'lane_departure': 0,
            'collision_warning': 0,
            'drowsiness': 0,
            'distraction': 0
        }
    
    def log_event(self, event_type, details, severity):
        """Log a significant event"""
        timestamp = datetime.datetime.now()
        
        # Update counter
        if event_type in self.event_counts:
            self.event_counts[event_type] += 1
        
        # Write to CSV
        with open(self.event_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, details, severity])
    
    def log_metric(self, metric_name, value):
        """Log a continuous metric value"""
        timestamp = datetime.datetime.now()
        
        # Write to CSV
        with open(self.metrics_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, metric_name, value])
    
    def save_screenshot(self, image, event_type):
        """Save a screenshot of an event"""
        screenshots_dir = os.path.join(self.log_dir, "screenshots")
        Path(screenshots_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event_type}_{timestamp}.png"
        filepath = os.path.join(screenshots_dir, filename)
        
        # Save image
        import cv2
        cv2.imwrite(filepath, image)
        return filepath
    
    def generate_report(self):
        """Generate a report of driving statistics"""
        report = {
            'session_start': self.timestamp,
            'session_end': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'event_counts': self.event_counts,
            'total_events': sum(self.event_counts.values()),
        }
        
        # Save report as JSON
        report_path = os.path.join(self.log_dir, f"report_{self.timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate charts
        self._generate_charts()
        
        return report_path
    
    def _generate_charts(self):
        """Generate charts for the report"""
        charts_dir = os.path.join(self.log_dir, "charts")
        Path(charts_dir).mkdir(parents=True, exist_ok=True)
        
        # Event count pie chart
        plt.figure(figsize=(10, 6))
        labels = list(self.event_counts.keys())
        sizes = list(self.event_counts.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Event Distribution')
        plt.savefig(os.path.join(charts_dir, f"event_distribution_{self.timestamp}.png"))
        plt.close()

# Example usage
if __name__ == "__main__":
    logger = ADASLogger()
    
    # Log some sample events
    logger.log_event("lane_departure", "Left lane departure detected", "WARNING")
    logger.log_event("collision_warning", "Vehicle ahead - distance: 15m", "CRITICAL")
    
    # Log some metrics
    logger.log_metric("vehicle_speed", 65.5)
    logger.log_metric("attention_score", 0.85)
    
    # Generate report
    report_path = logger.generate_report()
    print(f"Report generated at: {report_path}")