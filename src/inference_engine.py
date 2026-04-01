import onnxruntime as ort
import numpy as np
import time
from transformers import AutoTokenizer

class OnDeviceLLMInferenceEngine:
    def __init__(self, model_path, model_id="distilgpt2", providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        print(f"Initializing ONNX Runtime using providers: {providers}")
        
        # Session options for graph optimizations during runtime
        sess_options = ort.SessionOptions()
        # Enable all available high-level graph optimizations within ONNX Runtime
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load model into ONNX Runtime Session
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Get input names expected by the ONNX graph
        # Generally, it's input_ids and attention_mask
        self.inputs = {inp.name: inp for inp in self.session.get_inputs()}
        self.input_name_ids = "input_ids" if "input_ids" in self.inputs else self.session.get_inputs()[0].name
        self.input_name_mask = "attention_mask" if "attention_mask" in self.inputs else self.session.get_inputs()[1].name
        
        # Output tensor containing logits
        self.output_name = self.session.get_outputs()[0].name

    def generate(self, prompt, max_new_tokens=20):
        # Encode initial prompt
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int32)
        attention_mask = inputs["attention_mask"].astype(np.int32)

        print(f"\nPrompt: '{prompt}'")
        print("Model output: ", end="", flush=True)

        start_time = time.time()
        
        # Extremely basic greedy generation loop for demonstration
        for _ in range(max_new_tokens):
            # ONNX Runtime inference step
            ort_inputs = {
                self.input_name_ids: input_ids,
                self.input_name_mask: attention_mask
            }
            logits = self.session.run([self.output_name], ort_inputs)[0]
            
            # Get next token (greedy decoding)
            # logits shape might be [batch, sequence_length, vocab_size]
            next_token_logits = logits[0, -1, :]
            next_token_id = np.argmax(next_token_logits)
            
            # Append to prompt
            input_ids = np.append(input_ids, [[next_token_id]], axis=1).astype(np.int32)
            attention_mask = np.append(attention_mask, [[1]], axis=1).astype(np.int32)
            
            # Yield token for streaming feel (though running synchronously here)
            decoded_token = self.tokenizer.decode([next_token_id])
            print(decoded_token, end="", flush=True)
            
            # End of sequence check
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        end_time = time.time()
        latency = end_time - start_time
        
        final_text = self.tokenizer.decode(input_ids[0])
        print(f"\n\nInference Latency: {latency:.4f} seconds "
              f"({max_new_tokens / latency:.2f} tokens/second)")
        
        return final_text, latency
