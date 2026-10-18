"""Replace CLIP's non-overlapping patch Conv with equivalent patch extraction + MatMul.

This removes the only convolution from this pinned model, avoiding cuDNN startup.
The original downloaded model remains unchanged. Every generated model is checked
and compared against the original on deterministic inputs before it is accepted.
"""
from pathlib import Path
import json
import os

import numpy as np


def optimize(source: Path, destination: Path):
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    model = onnx.load(source)
    convolutions = [node for node in model.graph.node if node.op_type == "Conv"]
    if len(convolutions) != 1:
        raise ValueError("Expected exactly one patch-projection convolution")
    node = convolutions[0]
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    weights = {w.name: numpy_helper.to_array(w) for w in model.graph.initializer}
    weight = weights[node.input[1]]
    if weight.shape != (768, 3, 32, 32) or len(node.input) != 2 or attrs.get("strides") != [32, 32] or attrs.get("pads", [0]*4) != [0]*4 or attrs.get("group", 1) != 1 or attrs.get("dilations", [1,1]) != [1,1]:
        raise ValueError("This model does not match the validated CLIP patch projection")
    prefix = "adaptive_patch_v1_"
    initializers = {
        "shape1": np.asarray([-1, 3, 7, 32, 7, 32], dtype=np.int64),
        "shape2": np.asarray([-1, 3072], dtype=np.int64),
        "shape3": np.asarray([-1, 7, 7, 768], dtype=np.int64),
        "weight": weight.reshape(768, 3072).T.copy(),
    }
    for name, value in initializers.items():
        model.graph.initializer.append(numpy_helper.from_array(value, prefix + name))
    replacements = [
        helper.make_node("Reshape", [node.input[0], prefix+"shape1"], [prefix+"split"]),
        helper.make_node("Transpose", [prefix+"split"], [prefix+"patches"], perm=[0,2,4,1,3,5]),
        helper.make_node("Reshape", [prefix+"patches", prefix+"shape2"], [prefix+"flat"]),
        helper.make_node("MatMul", [prefix+"flat", prefix+"weight"], [prefix+"projected"]),
        helper.make_node("Reshape", [prefix+"projected", prefix+"shape3"], [prefix+"grid"]),
        helper.make_node("Transpose", [prefix+"grid"], list(node.output), perm=[0,3,1,2]),
    ]
    nodes = list(model.graph.node)
    index = nodes.index(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes[:index] + replacements + nodes[index+1:])
    remaining = [w for w in model.graph.initializer if w.name != node.input[1]]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining)
    onnx.checker.check_model(model)
    temporary = destination.with_suffix('.tmp.onnx')
    onnx.save(model, temporary)
    del model, weights
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    original = ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
    converted = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(20260926)
    maximum_error = 0
    for batch in (1, 3):
        pixels = generator.normal(0, 1, (batch, 3, 224, 224)).astype(np.float32)
        expected = original.run(None, {original.get_inputs()[0].name: pixels})
        actual = converted.run(None, {converted.get_inputs()[0].name: pixels})
        for a, b in zip(expected, actual):
            maximum_error = max(maximum_error, float(np.max(np.abs(a-b))))
            np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5)
    del original, converted
    os.replace(temporary, destination)
    report = {"transformation": "CLIP non-overlapping patch Conv -> equivalent reshape/transpose/MatMul",
              "validation_batches": [1,3], "seed": 20260926, "maximum_absolute_output_error": maximum_error,
              "validation_provider": "CPUExecutionProvider", "weights_changed": False}
    destination.with_suffix('.validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    from model_setup import prepare_environment
    root = prepare_environment() / 'clip/onnx'
    print(json.dumps(optimize(root/'vision_model.onnx', root/'vision_model_matmul.onnx'), indent=2))
