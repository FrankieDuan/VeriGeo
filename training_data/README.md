# Training Data

```
training_data/
├── train.jsonl
├── images/
└── README.md
```

## Record Schema

```json
{
  "format":    "numeric" | "mcq" | "mcq_zh",
  "question":  "...",
  "image":     "images/img_NNNNN.png",
  "answer":    "...",
  "rationale": "..."
}
```
