# On-Device LLM Inference Pipeline

A professional-grade inference pipeline demonstrating how to deploy transformer-based Large Language Models (LLMs) on edge hardware utilizing Python, TensorFlow, and ONNX Runtime.

## Overview
This repository provides a complete workflow for loading an autoregressive transformer, exporting its computation graph to `.onnx` format, and optimizing graph execution to significantly reduce edge hardware latency and memory layout.

### Key Capabilities:
- **Graph Tracing**: Leverage `@tf.function` endpoints to convert Hugging Face PyTorch/TensorFlow weights into an operational ONNX signature.
- **Dynamic Quantization**: Uses on-the-fly INT8 symmetric quantization to shrink dense layer footprints by ~75%.
- **Edge Inference Serving**: Replaces standard transformers library inference with lightning-fast ONNX ExecutionProviders.

## Project Structure
```text
on_device_llm_pipeline/
├── src/
│   ├── export_model.py     # Traces & exports HuggingFace models to ONNX
│   ├── optimize_graph.py   # Applies INT8 Dynamic Quantization 
│   ├── inference_engine.py # The ONNX Runtime execution engine wrapper
├── main.py                 # Pipeline orchestrator & benchmarking tool
├── requirements.txt        # Python dependency definitions
└── .gitignore              # Standard git exclusions
```

## Setup & Execution
It is recommended to run this within a dedicated virtual environment due to the large binary dependencies of TF and ONNX.

```bash
# Clone the repository
git clone https://github.com/Ramya-Pravallika/ondevice_llm_pipeline.git
cd ondevice_llm_pipeline

# Configure Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Execute End-to-End Pipeline & Benchmarks
python main.py
```
