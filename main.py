import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.export_model import export_huggingface_to_onnx
from src.optimize_graph import optimize_and_quantize_model
from src.inference_engine import OnDeviceLLMInferenceEngine

def main():
    model_id = "distilgpt2"
    
    # We use a single prompt for fair benchmarking
    prompt = "The future of artificial intelligence in edge devices is"
    max_tokens = 15
    
    # Directories setup
    os.makedirs("models", exist_ok=True)
    base_model_path = "models/distilgpt2_base.onnx"
    quantized_model_path = "models/distilgpt2_quantized.onnx"
    
    print("\n" + "="*60)
    print("STEP 1: Load and Export Model to ONNX Graph")
    print("="*60)
    if not os.path.exists(base_model_path):
        export_huggingface_to_onnx(model_id, base_model_path)
    else:
        print(f"Model already exported at {base_model_path}")
        
    print("\n" + "="*60)
    print("STEP 2: Optimize and Quantize (INT8) Graph for Edge")
    print("="*60)
    if not os.path.exists(quantized_model_path):
        optimize_and_quantize_model(base_model_path, quantized_model_path)
    else:
        print(f"Quantized model already available at {quantized_model_path}")
        
    print("\n" + "="*60)
    print("STEP 3: Validate Inference Engine (Unoptimized)")
    print("="*60)
    engine_base = OnDeviceLLMInferenceEngine(base_model_path, model_id)
    _, latency_base = engine_base.generate(prompt, max_new_tokens=max_tokens)
    
    print("\n" + "="*60)
    print("STEP 4: Validate Inference Engine (Optimized & Quantized)")
    print("="*60)
    engine_quant = OnDeviceLLMInferenceEngine(quantized_model_path, model_id)
    _, latency_quant = engine_quant.generate(prompt, max_new_tokens=max_tokens)
    
    print("\n" + "="*60)
    print("SUMMARY: Benchmark Comparison")
    print("="*60)
    print(f"Base Model Latency:      {latency_base:.4f} s")
    print(f"Quantized Model Latency: {latency_quant:.4f} s")
    if latency_base > 0:
        speedup = (latency_base - latency_quant) / latency_base * 100
        print(f"Latency improvement via Optimization: {speedup:.2f}%")

if __name__ == '__main__':
    main()
