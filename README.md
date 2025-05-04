# Smart Traffic Light Control System

## Overview

The Smart Traffic Light Control System is a neuro-fuzzy model designed to optimize green light durations at urban intersections, reducing vehicle waiting times. It integrates fuzzy logic for rule-based reasoning and a neural network for data-driven predictions, using six input parameters:

- **Vehicle Density**: Number of vehicles (0–100).
- **Waiting Time**: Average wait time (0–60 seconds).
- **Traffic Flow Rate**: Vehicles per minute (0–60).
- **Queue Length**: Queued vehicles (0–30).
- **Emergency Vehicle Presence**: Binary (0: absent, 1: present).
- **Opposing Traffic Density**: Vehicles in opposing lanes (0–100).

The system processes a 10,000-sample dataset (`traffic_data_10000.csv`) and evaluates performance across subsets (100, 500, 1000, 2500, 5000, 7500, 10000 samples), producing metrics, test results, and visualizations.

---

## Dataset

The dataset (`traffic_data_10000.csv`) simulates urban traffic across three scenarios:

1. **Rush Hour (30%)**:
   - High density (70–100), long waits (40–60s), high flow (40–60), long queues (20–30), 5% emergency chance.
2. **Off-Peak (50%)**:
   - Medium density (20–70), medium waits (10–40s), medium flow (20–40), medium queues (5–20), 2% emergency chance.
3. **Nighttime (20%)**:
   - Low density (0–20), short waits (0–15s), low flow (0–20), short queues (0–10), 1% emergency chance.

**Details**:

- **Columns**: `vehicle_density`, `waiting_time`, `flow_rate`, `queue_length`, `emergency_vehicle`, `opposing_density`, `green_duration`.
- **Size**: 10,000 samples.
- **Split**: 80% training, 20% testing.
- **Random Seed**: 42 for reproducibility.

---

## System Components

### Fuzzy Logic System

Built with `scikit-fuzzy`, the fuzzy system uses 9 rules to map inputs to green durations (10–60 seconds).

#### Fuzzy Variables

**Inputs**:

- `vehicle_density`: Low (0–50), Medium (25–75), High (50–100).
- `waiting_time`: Short (0–30s), Medium (15–45s), Long (30–60s).
- `flow_rate`: Low (0–25), Medium (15–55), High (45–60).
- `queue_length`: Short (0–12), Medium (8–28), Long (22–30).
- `emergency_vehicle`: Absent (0), Present (1).
- `opposing_density`: Low (0–50), Medium (25–75), High (50–100).

**Output**:

- `green_duration`: Short (10–30s), Medium (20–50s), Long (40–60s).

**Membership Functions**: Triangular (`trimf`).

#### Fuzzy Rules

Key rules include:

1. High density, long wait, high flow, long queue, no emergency, low opposing → **Long green**.
2. Medium density, medium wait, medium flow, medium queue, no emergency → **Medium green**.
3. Low density, short wait, low flow, short queue, no emergency, high opposing → **Short green**.
4. Emergency vehicle present → **Long green**.
5. High flow, long queue, low opposing → **Long green**.
6. High opposing density, low density → **Short green**.
7. High density, short queue, medium flow → **Medium green**.
8. Long wait, low flow, medium queue → **Medium green**.
9. Low density, low opposing, no emergency → **Short green**.

---

### Neural Network

- **Model**: `MLPRegressor` (scikit-learn), 2 hidden layers (10 neurons each), 1000 iterations.
- **Training**: On 80% of dataset, predicting green durations.
- **Output**: Combined with fuzzy output (70% fuzzy, 30% neural), clipped to 10–60 seconds.

---

### Integration

For each test sample:

1. Compute fuzzy output using `scikit-fuzzy`.
2. Predict neural output using `MLPRegressor`.
3. Combine outputs (weighted average).
4. Calculate wait time: `max(0, waiting_time - green_time)`.

---

## Evaluation

The system is evaluated across sample sizes: 100, 500, 1000, 2500, 5000, 7500, 10000.

### Metrics

- **Average Waiting Time**: Mean wait time on test set (seconds).
- **Training Time**: Neural network training duration (seconds).

### Outputs

- `performance_metrics.csv`: Metrics per sample size.
- `test_results_10000.csv`: Test results (~2000 samples) with inputs, predicted green durations, and wait times.
- `traffic_results_10000.txt`: Summary of avg wait time, train/test split, and sample data.

---

### Example Performance Metrics

| Sample Size | Avg Wait Time (s) | Training Time (s) |
|-------------|-------------------|-------------------|
| 100         | 14.79            | 0.63              |
| 500         | 15.12            | 0.78              |
| 1000        | 15.27            | 0.92              |
| 2500        | 15.31            | 1.15              |
| 5000        | 15.29            | 1.42              |
| 7500        | 15.30            | 1.68              |
| 10000       | 15.29            | 1.95              |

---

## Visualizations

The notebook generates:

1. **Performance Plot (`performance_plot.png`)**:
   - Left: Waiting time vs. sample size.
   - Right: Training time vs. sample size.

2. **Traffic Plots (`traffic_plot_<sample_size>.png`)**:
   - Scatter plots of vehicle density vs. waiting time, colored by green duration (viridis colormap).
   - Files: `traffic_plot_100.png`, `traffic_plot_500.png`, ..., `traffic_plot_10000.png`.

---

## File Structure

| File                          | Description                                           |
|-------------------------------|-------------------------------------------------------|
| `traffic_control_system.ipynb`| Jupyter Notebook with all code (data generation, model, evaluation, plots). |
| `traffic_data_10000.csv`      | 10,000-sample dataset.                                |
| `performance_metrics.csv`     | Performance metrics across sample sizes.             |
| `test_results_10000.csv`      | Test results for 10,000-sample evaluation (~2000 samples). |
| `traffic_results_10000.txt`   | Summary of 10,000-sample evaluation.                 |
| `performance_plot.png`        | Plot of waiting time and training time vs. sample size. |
| `traffic_plot_*.png`          | Scatter plots for each sample size (100, 500, ..., 10000). |

---

## Setup and Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. **Install Python 3.12**: Use a Python 3.12 environment (e.g., via pyenv or conda).

3. **Install Dependencies**:

    ```bash
        pip install scikit-fuzzy numpy scipy networkx matplotlib scikit-learn pandas


