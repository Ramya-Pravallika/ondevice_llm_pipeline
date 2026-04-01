import os
import tensorflow as tf
import tf2onnx
from transformers import AutoTokenizer, TFAutoModelForCausalLM

def export_huggingface_to_onnx(model_id="distilgpt2", output_path="models/model.onnx"):
    """
    Downloads a HuggingFace Transformer model and traces it to an ONNX graph.
    """
    print(f"Downloading tokenizer and TF model for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = TFAutoModelForCausalLM.from_pretrained(model_id)

    # Define the input signature expected by the model during inference.
    # DistilGPT2 and similar models use input_ids and attention_mask.
    input_signature = [
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="attention_mask")
    ]

    print(f"Tracing model for ONNX backend...")
    @tf.function(input_signature=input_signature)
    def serving_fn(input_ids, attention_mask):
        # We only return logits for inference speed and to reduce graph size
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    print(f"Converting TF model to ONNX graph...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Exporting the traced function to standard ONNX format using opset 15
    model_proto, _ = tf2onnx.convert.from_function(
        serving_fn, 
        input_signature=input_signature, 
        opset=15, 
        output_path=output_path
    )
    
    print(f"Successfully exported ONNX model to {output_path}")
    return output_path

if __name__ == '__main__':
    export_huggingface_to_onnx()
