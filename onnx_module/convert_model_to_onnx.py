import argparse
import os
from pathlib import Path

from psutil import cpu_count

from const import ROOT_DIR

os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
from transformers.convert_graph_to_onnx import convert
from transformers import AutoTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from transformers import BertTokenizerFast
from transformers.convert_graph_to_onnx import quantize
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertModel
from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
from utils import set_seed, dotdict


@dataclass
class OnnxInferenceResult:
    model_inference_time: [int]
    optimized_model_path: str


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model_name = args.model_name
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_name)
    num_hidden_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    onnx_model_path = "onnx/bert-base-cased.onnx"
    convert(framework="pt", model=model_name, output=Path(onnx_model_path), opset=11)
    opt_options = BertOptimizationOptions('bert')
    opt_options.enable_embed_layer_norm = False
    opt_model = optimizer.optimize_model(
        onnx_model_path,
        'bert',
        num_heads=num_attention_heads,
        hidden_size=hidden_size,
        optimization_options=opt_options)
    opt_model.save_model_to_file('bert.opt.onnx')

    # Constants from the performance optimization available in onnxruntime
    # It needs to be done before importing onnxruntime

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    cpu_model = create_model_for_provider(onnx_model_path, "CPUExecutionProvider")

    # Inputs are provided through numpy array
    model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
    # Run the model (None = get all the outputs)
    sequence, pooled = cpu_model.run(None, inputs_onnx)
    # Print information about outputs
    print(f"Sequence output: {sequence.shape}, Pooled output: {pooled.shape}")

    PROVIDERS = {
        ("cpu", "PyTorch CPU"),
        #  Uncomment this line to enable GPU benchmarking
        #    ("cuda:0", "PyTorch GPU")
    }

    results = {}

    for device, label in PROVIDERS:

        # Move inputs to the correct device
        model_inputs_on_device = {
            arg_name: tensor.to(device)
            for arg_name, tensor in model_inputs.items()
        }

        # Add PyTorch to the providers
        model_pt = BertModel.from_pretrained("bert-base-cased").to(device)
        for _ in trange(10, desc="Warming up"):
            model_pt(**model_inputs_on_device)

        # Compute
        time_buffer = []
        for _ in trange(100, desc=f"Tracking inference time on PyTorch"):
            with track_infer_time(time_buffer):
                model_pt(**model_inputs_on_device)

        # Store the result
        results[label] = OnnxInferenceResult(
            time_buffer,
            None
        )

    PROVIDERS = {
        ("CPUExecutionProvider", "ONNX CPU"),
        #  Uncomment this line to enable GPU benchmarking
        #     ("CUDAExecutionProvider", "ONNX GPU")
    }

    for provider, label in PROVIDERS:
        # Create the model with the specified provider
        model = create_model_for_provider("onnx/bert-base-cased.onnx", provider)

        # Keep track of the inference time
        time_buffer = []

        # Warm up the model
        model.run(None, inputs_onnx)

        # Compute
        for _ in trange(100, desc=f"Tracking inference time on {provider}"):
            with track_infer_time(time_buffer):
                model.run(None, inputs_onnx)

        # Store the result
        results[label] = OnnxInferenceResult(
            time_buffer,
            model.get_session_options().optimized_model_filepath
        )

    # Commented out IPython magic to ensure Python compatibility.
    # %matplotlib inline

    # Compute average inference time + std
    time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
    time_results_std = np.std([v.model_inference_time for v in results.values()]) * 1000

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylabel("Avg Inference time (ms)")
    ax.set_title("Average inference time (ms) for each provider")
    ax.bar(time_results.keys(), time_results.values(), yerr=time_results_std)
    plt.show()

    # Quantize
    model_pt_quantized = torch.quantization.quantize_dynamic(
        model_pt.to("cpu"), {torch.nn.Linear}, dtype=torch.qint8
    )

    # Warm up
    model_pt_quantized(**model_inputs)

    # Benchmark PyTorch quantized model
    time_buffer = []
    for _ in trange(100):
        with track_infer_time(time_buffer):
            model_pt_quantized(**model_inputs)

    results["PyTorch CPU Quantized"] = OnnxInferenceResult(
        time_buffer,
        None
    )

    # Transformers allow you to easily convert float32 model to quantized int8 with ONNX Runtime
    quantized_model_path = quantize(Path("bert.opt.onnx"))

    # Then you just have to load through ONNX runtime as you would normally do
    quantized_model = create_model_for_provider(quantized_model_path.as_posix(), "CPUExecutionProvider")

    # Warm up the overall model to have a fair comparaison
    outputs = quantized_model.run(None, inputs_onnx)

    # Evaluate performances
    time_buffer = []
    for _ in trange(100, desc=f"Tracking inference time on CPUExecutionProvider with quantized model"):
        with track_infer_time(time_buffer):
            outputs = quantized_model.run(None, inputs_onnx)

    # Store the result
    results["ONNX CPU Quantized"] = OnnxInferenceResult(
        time_buffer,
        quantized_model_path
    )

    """## Show the inference performance of each providers """
    # Compute average inference time + std
    time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
    time_results_std = np.std([v.model_inference_time for v in results.values()]) * 1000

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylabel("Avg Inference time (ms)")
    ax.set_title("Average inference time (ms) for each provider")
    ax.bar(time_results.keys(), time_results.values(), yerr=time_results_std)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning bert')
    parser.add_argument("--model_name", default="", type=str, required=False,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--tokenizer_name",
                        default="bert-base-uncased",
                        type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",
                        required=False)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--output_dir", default=str(ROOT_DIR/"models"/"onnx"), type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()
    args = vars(args)
    args = dotdict(args)
    set_seed(args.seed)
    main(args)
