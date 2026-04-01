import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize_and_quantize_model(input_model_path="models/model.onnx", output_model_path="models/model_quantized.onnx"):
    """
    Applies ONNX Dynamic Quantization (to INT8) and optional basic optimizations
    to the input model, outputting to output_model_path.
    """
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model not found at {input_model_path}")
    
    print(f"Applying INT8 Dynamic Quantization to {input_model_path}...")
    
    # Quantize the model using dynamic quantization:
    # This is well suited for language models (Transformer variants) because 
    # it primarily targets the dense feed-forward & attention layers.
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8,
        # optimize_model=True applies constant folding and other graph optimizations automatically.
        optimize_model=True 
    )
    
    # Calculate file size reduction
    orig_size = os.path.getsize(input_model_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_model_path) / (1024 * 1024)
    
    print(f"Optimization complete. Saved optimized graph to {output_model_path}")
    print(f"Original Model Size:  {orig_size:.2f} MB")
    print(f"Quantized Model Size: {quant_size:.2f} MB")
    print(f"Total Size Reduction: {(1 - quant_size/orig_size)*100:.2f}%")
    
    return output_model_path

if __name__ == '__main__':
    optimize_and_quantize_model()
