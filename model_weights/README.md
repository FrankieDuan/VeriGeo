# Model Weights

LoRA adapter for `Qwen/Qwen2.5-VL-7B-Instruct`.

```python
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

base = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="bfloat16")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = PeftModel.from_pretrained(base, "./model_weights")
model.eval()
```
