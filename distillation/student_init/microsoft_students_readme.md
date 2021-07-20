
You can use the following *task-agnostic pre-distilled checkpoints* from XtremeDistilTransformers for (only) fine-tuning on labeled data from downstream tasks:
- [6/256 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l6-h256-uncased)
- [6/384 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l6-h384-uncased)
- [12/384 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l12-h384-uncased)

For further performance improvement, initialize XtremeDistilTransformers with any of the above pre-distilled checkpoints for *task-specific distillation* with additional unlabeled data from the downstream task for the best performance.

The following table shows the performance of the above checkpoints on GLUE dev set and SQuAD-v2.

| Models         | #Params | Speedup | MNLI | QNLI | QQP  | RTE  | SST  | MRPC | SQUAD2 | Avg   |
|----------------|--------|---------|------|------|------|------|------|------|--------|-------|
| BERT        | 109    | 1x       | 84.5 | 91.7 | 91.3 | 68.6 | 93.2 | 87.3 | 76.8   | 84.8 |
| DistilBERT  | 66     | 2x       | 82.2 | 89.2 | 88.5 | 59.9 | 91.3 | 87.5 | 70.7   | 81.3 |
| TinyBERT    | 66     | 2x       | 83.5 | 90.5 | 90.6 | 72.2 | 91.6 | 88.4 | 73.1   | 84.3 |
| MiniLM      | 66     | 2x       | 84.0   | 91.0   | 91.0   | 71.5 | 92.0   | 88.4 | 76.4   | 84.9  |
| MiniLM      | 22     | 5.3x     | 82.8 | 90.3 | 90.6 | 68.9 | 91.3 | 86.6 | 72.9   | 83.3 |
| XtremeDistil-l6-h256   | 13     | 8.7x     | 83.9 | 89.5 | 90.6   | 80.1 | 91.2 | 90.0   | 74.1   | 85.6 |
| XtremeDistil-l6-h384   | 22     | 5.3x     | 85.4 | 90.3 | 91.0   | 80.9 | 92.3 | 90.0   | 76.6   | 86.6 |
| XtremeDistil-l12-h384   | 33     | 2.7x     | 87.2 | 91.9 | 91.3   | 85.6 | 93.1 | 90.4   | 80.2   | 88.5 |