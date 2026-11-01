# Multimodal Cross-Attention Network for Sensor Fusion

This project demonstrates how to fuse two complementary sensor streams for anomaly detection using a dual 1D-CNN backbone and a cross-attention mechanism. The goal is to model the real-world challenge of combining multiple sensor views that carry different but related information about the same system.

## Why this project matters

Modern monitoring systems often rely on multiple sensors that observe the same process from different perspectives: timing signals, structural measurements, vibration, temperature, electrical output, or other stream-based observations. These modalities are not independent. They often contain correlated patterns that reveal faults, anomalies, or state changes earlier than any single sensor can detect.

The core problem this project addresses is sensor fusion under real-world conditions where signals can be noisy, partially missing, distorted, or biased by instrumentation effects. Instead of treating each stream separately, the model learns how to align and compare them, so it can focus on the most relevant relationships across time and feature space.

This is useful in domains such as:

- industrial monitoring and predictive maintenance
- scientific instrumentation and observational systems
- aerospace and satellite telemetry
- medical signal analysis
- autonomous sensing and robotics

## What the model does

The notebook builds a multimodal architecture that:

- preprocesses paired time-series streams
- fills missing or interrupted segments
- normalizes feature scales for stable training
- extracts features from each stream with separate 1D CNN encoders
- uses cross-attention to align the streams and emphasize their meaningful interactions
- classifies the fused representation into normal vs anomalous behavior

The attention layer is especially important because it allows the model to explain which parts of one sensor stream are most relevant when interpreting the other. This helps not only with accuracy, but also with understanding how the system is making decisions.

## Repository contents

- `multimodal_cross_attention_sensor_fusion.ipynb` — the full end-to-end experiment
- `requirements.txt` — Python dependencies

## Getting started

Use Python 3.10 or newer, then install the dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `multimodal_cross_attention_sensor_fusion.ipynb` in JupyterLab or VS Code and run the cells in order.

## Real-world impact

This project is a practical example of multimodal deep learning for monitoring and detection. Rather than relying on one noisy or incomplete signal, it learns to merge multiple sensor representations into a more robust understanding of system behavior. In real systems, this can improve early anomaly detection, reduce false alarms, and support more reliable decision-making in complex environments.
