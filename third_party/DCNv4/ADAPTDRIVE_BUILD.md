# AdaptDrive DCNv4 build record

The validated AdaptDrive environment uses:

```text
Python: 3.8.20
PyTorch: 1.13.0+cu117
CUDA toolkit: 11.8
GPU used for validation: NVIDIA GeForce RTX 4090 (physical GPU 3)
DCNv4: 1.0.0.post2 at commit 4b848f7dd7da74ff03f7d278f902c6fd05b391b5
```

PyTorch 1.13 does not recognize the explicit Ada architecture label `8.9` in `TORCH_CUDA_ARCH_LIST`. The validated build therefore uses `8.6+PTX`: it emits an Ampere cubin and PTX that the NVIDIA driver JIT-compiles for the RTX 4090. This avoids modifying upstream DCNv4 or PyTorch build helpers.

Build from a temporary copy so generated products do not enter the source tree:

```bash
BUILD_ROOT="$(mktemp -d /tmp/adaptdrive-dcnv4-build.XXXXXX)"
cp -a third_party/DCNv4/. "${BUILD_ROOT}/"
cd "${BUILD_ROOT}/DCNv4_op"
CUDA_VISIBLE_DEVICES=<physical_gpu> \
CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST='8.6+PTX' \
MAX_JOBS=4 \
python setup.py bdist_wheel
python -m pip install --force-reinstall --no-deps dist/DCNv4-1.0.0.post2-*.whl
```

The August 22, 2026 validated wheel and installed extension were:

```text
wheel_sha256: ffa7b59cca0c66dbb1a7e426b2b4f891d24279ac5ea9b3d34d9e808c95ed2a4e
extension_sha256: 18a608cc227cea335119cbe6625e7b1d4e0f7c8a87630618803adf669956933b
```

The wheel is a machine/environment build product and is not part of the canonical source snapshot. After installation, run `Bench2Drive/test_feature_dcnv4_adapter_smoke.py` on an approved GPU and require both non-zero feature gradients and non-zero DCNv4 parameter gradients.
