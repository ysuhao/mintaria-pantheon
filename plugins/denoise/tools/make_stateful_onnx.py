#!/usr/bin/env python3
"""把官方 DeepFilterNet3 的三个 ONNX 改成可逐帧调用的有状态版本。

官方导出的图按整段推理设计：GRU 的初始状态是常量 0，时间卷积前用 Pad 补零。
ONNX Runtime 没有 tract 那样的 pulse 变换，所以这里做最小的图手术，权重一个字节都不改：

- 每个 GRU 的 initial_h 换成图输入 h_<n>，Y_h 接到图输出 h_<n>_out。
- 每个时间轴 Pad 换成 Concat(p_<n>, x)，x 接到图输出 p_<n>_out。
  p_<n> 由调用方维护，存放该卷积需要的历史帧，帧数由输入形状给出。
- 固定输入形状，方便 ONNX Runtime 做常量折叠，也让 C++ 端能从模型读出状态的形状。

用法：python -I make_stateful_onnx.py <官方模型目录> <输出目录>
"""
import sys
from pathlib import Path

import onnx
from onnx import helper

EMB_SIZE = 512  # DF3 与 DF3-LL 的 emb 都是 conv_ch * nb_erb / 4

# 单帧张量的形状，时间轴固定为第 2 维。名字来自官方图。
FRAME_SHAPES = {
    "enc": {"feat_erb": [1, 1, 1, 32], "feat_spec": [1, 2, 1, 96],
            "e0": [1, 64, 1, 32], "e1": [1, 64, 1, 16], "e2": [1, 64, 1, 8], "c0": [1, 64, 1, 96]},
    "erb_dec": {"emb": [1, 1, EMB_SIZE], "e3": [1, 64, 1, 8], "e2": [1, 64, 1, 8], "e1": [1, 64, 1, 16], "e0": [1, 64, 1, 32],
                "/Add_output_0": [1, 64, 1, 8], "/Add_3_output_0": [1, 64, 1, 32]},
    "df_dec": {"emb": [1, 1, EMB_SIZE], "c0": [1, 64, 1, 96]},
}


def set_shape(graph, name, dims):
    for value in graph.input:
        if value.name == name:
            shape = value.type.tensor_type.shape
            del shape.dim[:]
            for d in dims:
                shape.dim.add().dim_value = d
            return
    raise KeyError(name)


def expose_gru_states(graph):
    grus = [n for n in graph.node if n.op_type == "GRU"]
    for i, gru in enumerate(grus):
        hidden = next(a.i for a in gru.attribute if a.name == "hidden_size")
        graph.input.append(helper.make_tensor_value_info(f"h_{i}", onnx.TensorProto.FLOAT, [1, 1, hidden]))
        gru.input[5] = f"h_{i}"
        graph.node.append(helper.make_node("Identity", [gru.output[1]], [f"h_{i}_out"], name=f"state_out_h{i}"))
        graph.output.append(helper.make_tensor_value_info(f"h_{i}_out", onnx.TensorProto.FLOAT, [1, 1, hidden]))
    return len(grus)


def replace_time_pads(graph, frame_shapes):
    pads = [n for n in graph.node if n.op_type == "Pad"]
    nodes = list(graph.node)
    for i, pad in enumerate(pads):
        conv = next(n for n in nodes if pad.output[0] in n.input)
        kernel = next(a.ints for a in conv.attribute if a.name == "kernel_shape")
        shape = list(frame_shapes[pad.input[0]])
        shape[2] = kernel[0] - 1

        graph.input.append(helper.make_tensor_value_info(f"p_{i}", onnx.TensorProto.FLOAT, shape))
        concat = helper.make_node("Concat", [f"p_{i}", pad.input[0]], [pad.output[0]], name=f"history_{i}", axis=2)
        graph.node.insert(list(graph.node).index(pad), concat)
        graph.node.remove(pad)
        graph.node.append(helper.make_node("Identity", [pad.input[0]], [f"p_{i}_out"], name=f"state_out_p{i}"))
        graph.output.append(helper.make_tensor_value_info(f"p_{i}_out", onnx.TensorProto.FLOAT, frame_shapes[pad.input[0]]))
    return len(pads)


def prune_dead_nodes(graph):
    """去掉手术后不再被用到的节点（原来的零状态和 Pad 形状计算）。"""
    needed = {o.name for o in graph.output}
    keep = []
    for node in reversed(list(graph.node)):
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
    keep.reverse()
    del graph.node[:]
    graph.node.extend(keep)
    live = [i for i in graph.initializer if i.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(live)


def convert(src, dst, name):
    model = onnx.load(src / f"{name}.onnx")
    graph = model.graph
    shapes = FRAME_SHAPES[name]

    n_gru = expose_gru_states(graph)
    n_pad = replace_time_pads(graph, shapes)
    prune_dead_nodes(graph)
    for value in list(graph.input):
        if value.name in shapes:
            set_shape(graph, value.name, shapes[value.name])
    # 旧 value_info 里的时间维是手术前的，清掉让 Runtime 重新推断。输出形状的 S 是符号，不用改。
    del graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, dst / f"{name}.onnx")
    print(f"{name}: {n_gru} GRU, {n_pad} Pad")


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("enc", "erb_dec", "df_dec"):
        convert(src, dst, name)


if __name__ == "__main__":
    main()
