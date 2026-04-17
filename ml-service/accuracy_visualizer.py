import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_accuracy_report():
    # 1. METRICS DATA (Based on your latest output)
    metrics = {
        'R2 Score': 0.8954,
        'Accuracy': 0.8088,
        'MAPE': 0.1912
    }
    
    # 2. CREATE A BAR COMPARISON CHART
    plt.figure(figsize=(10, 6))
    names = list(metrics.keys())
    values = [v * 100 for v in metrics.values()] # Convert to percentages
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = plt.bar(names, values, color=colors)
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', fontweight='bold')

    plt.ylim(0, 100)
    plt.title('AePDS Model Performance Summary', fontsize=15)
    plt.ylabel('Percentage (%)')
    plt.savefig('model_metrics_summary.png')
    print("Saved: model_metrics_summary.png")

    # 3. CREATE ERROR MARGIN PIE (MAPE)
    plt.figure(figsize=(8, 8))
    plt.pie([80.88, 19.12], labels=['Correct Prediction', 'Error Margin (MAPE)'], 
            autopct='%1.1f%%', colors=['#2ecc71', '#ecf0f1'], startangle=140, explode=(0.1, 0))
    plt.title('Prediction Confidence vs Error')
    plt.savefig('error_margin_pie.png')
    print("Saved: error_margin_pie.png")

if __name__ == "__main__":
    create_accuracy_report()