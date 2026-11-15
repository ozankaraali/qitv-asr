"""The quantized Moonshine graph's int8 MatMulInteger weights, and their zero points, as uint8 (+128 each).

(B - zero point) is unchanged, so every product is the same, but onnxruntime then takes its u8u8 kernel. Its u8s8
kernel on x86 processors without VNNI (AVX2 only: most laptops before 2019) adds pairs of products in 16 bits and
saturates when per-channel weights use their full range: the per-channel decoder then repeats one word ("Merhaba,
Merhaba, Merhaba") or counts. Processors with VNNI, and ARM, read both forms alike. The released decoders were made
with this step (the encoders are uint8 already, and are left as they are).
usage: python moon_u8.py <in.onnx> <out.onnx>   (in place when both are the same file)"""
import shutil
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)

graphs = []


def collect(graph):
    graphs.append(graph)
    for node in graph.node:
        for a in node.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                collect(a.g)


collect(model.graph)
inits = {}
for g in graphs:
    for i in g.initializer:
        inits.setdefault(i.name, i)

uses = defaultdict(list)
for g in graphs:
    for node in g.node:
        for k, name in enumerate(node.input):
            if name in inits:
                uses[name].append((node.op_type, k))

targets = set()
for g in graphs:
    for node in g.node:
        if node.op_type == "MatMulInteger":
            for k in (1, 3):
                if len(node.input) > k and node.input[k] in inits:
                    if inits[node.input[k]].data_type == onnx.TensorProto.INT8:
                        targets.add(node.input[k])

if not targets:  # an encoder (uint8 already): the file as it is
    if src != dst:
        shutil.copyfile(src, dst)
    print(f"{src}: no int8 weights")
    sys.exit(0)
shared = [n for n in targets if any(u not in (("MatMulInteger", 1), ("MatMulInteger", 3)) for u in uses[n])]
if shared:
    raise SystemExit(f"int8 initializers used elsewhere too: {shared[:5]}")
for g in graphs:
    for node in g.node:
        if node.op_type == "MatMulInteger" and node.input[1] in targets:
            if len(node.input) < 4 or node.input[3] not in targets:
                raise SystemExit(f"{node.name}: an int8 weight without an int8 zero point")

for name in sorted(targets):
    values = numpy_helper.to_array(inits[name]).astype(np.int16) + 128
    inits[name].CopyFrom(numpy_helper.from_array(values.astype(np.uint8), name))
print(f"{src}: {len(targets)} int8 initializers made uint8")
onnx.checker.check_model(model)
onnx.save(model, dst)
