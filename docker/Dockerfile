# edge0-vllm:dev = the NGC vLLM image + the edge0-vllm plugin (DESIGN.md 2.10). Nothing else in vLLM changes.
#
#   cd /home/andychen/work/edge0-vllm && nice -n 19 docker build -f docker/Dockerfile -t edge0-vllm:dev .
#
# The build context is the edge0 tree; docker/Dockerfile.dockerignore (BuildKit) limits it to plugin/.
# For iteration without rebuilding, serve.sh EDGE0_DEV=1 bind-mounts the source over the installed copy
# (PYTHONPATH=/workspace/edge0/plugin); the installed dist-info keeps providing the vllm.general_plugins entry point.
FROM nvcr.io/nvidia/vllm:26.11-py3

COPY plugin /opt/edge0/plugin
# --no-deps: never let pip touch torch / vllm / flashinfer (PIP_CONSTRAINT stays in force for everything else).
RUN pip install --no-deps --no-build-isolation --no-cache-dir /opt/edge0/plugin \
 && python -c "from importlib.metadata import entry_points as ep; \
e = [x for x in ep(group='vllm.general_plugins') if x.name == 'edge0']; \
assert e and e[0].value == 'edge0_vllm:register', e; print('edge0 plugin entry point:', e[0])"

# EDGE0_CPU_EXPERTS (docs/CPU-EXPERTS.md) and EDGE0_PLE_DEFER (ple_defer.py): prebuild the C++/CUDA extensions (sm_120)
# into /opt/edge0/ext/<source hash>; cpu_experts.load() finds it there. A dev-mode source tree (EDGE0_DEV=1) has
# another hash and builds once into the mounted cache (/root/.cache/edge0-vllm/ext).
RUN cd /tmp && python -c "import os; from edge0_vllm import cpu_experts as c; \
d = os.path.join('/opt/edge0/ext', c.ext_key()); c.load(build_dir=d); print('edge0 cpu-experts extension:', d)" \
 && python -c "import os; from edge0_vllm import ple_defer as p; \
d = os.path.join('/opt/edge0/ext', p.ext_key()); p.load(build_dir=d); print('edge0 ple-defer extension:', d)" \
 && rm -rf /opt/edge0/ext/*/*.o /opt/edge0/ext/*/.ninja*

LABEL org.opencontainers.image.title="edge0-vllm" \
      org.opencontainers.image.description="nvcr.io/nvidia/vllm:26.11-py3 + edge0-vllm plugin (Edge0-145B-NVFP4)"
