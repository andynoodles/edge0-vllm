"""Dump GGUF metadata (scalars and short arrays) and tensor infos for a few blocks. Header only, no tensor data read.

docker run --rm --memory 3g --cpus 2 -v /home/andychen/models:/models:ro \
    -v /home/andychen/work/edge0/edge0-strata/third_party/llama.cpp/gguf-py:/gguf-py:ro \
    -v /home/andychen/work/edge0-vllm:/workspace/edge0 --entrypoint bash nvcr.io/nvidia/vllm:26.11-py3 \
    -c 'PYTHONPATH=/gguf-py nice -n 19 python /workspace/edge0/tools/gguf_meta_dump.py /models/Edge0-145B-NVFP4/model-skeleton.gguf 0,1,2,3,4,47'
"""
import re
import sys

from gguf import GGUFReader, GGUFValueType

path = sys.argv[1]
blocks = {int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3").split(",")}
r = GGUFReader(path, "r")
print(f"== {path}: {len(r.fields)} fields, {len(r.tensors)} tensors")
for name, f in r.fields.items():
    if name.startswith("GGUF."):
        continue
    t = f.types[0] if f.types else None
    if t == GGUFValueType.ARRAY:
        n = len(f.data)
        if n > 64:
            print(f"{name} = <array of {n} {f.types[-1].name}>")
            continue
        try:
            v = f.contents()
        except Exception:  # noqa: BLE001
            v = "?"
        print(f"{name} = {v}")
    else:
        try:
            v = f.contents()
        except Exception:  # noqa: BLE001
            v = "?"
        if isinstance(v, str) and len(v) > 200:
            v = v[:200] + "..."
        print(f"{name} = {v}")
print("== tensors (non-block + selected blocks)")
seen_other = 0
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.", t.name)
    if m and int(m.group(1)) not in blocks:
        continue
    if not m:
        seen_other += 1
    print(f"{t.name:60s} {t.tensor_type.name:8s} shape={list(t.shape)} bytes={int(t.n_bytes)} off={int(t.data_offset)}")
