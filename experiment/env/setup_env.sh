#!/usr/bin/env bash
set -ex
exec > >(tee -a /root/ebgrpo/env_create.log) 2>&1
PIP_TIMEOUT="--timeout 30 --retries 3"
TENCENT=https://mirrors.cloud.tencent.com/pypi/simple
ALIYUN=https://mirrors.aliyun.com/pypi/simple
source /opt/conda/etc/profile.d/conda.sh
/opt/conda/bin/conda env list | grep -q '^ebgrpo ' || /opt/conda/bin/conda create -n ebgrpo python=3.10 -y
conda activate ebgrpo
if ! python -c "import vllm" 2>/dev/null; then
  pip install $PIP_TIMEOUT \
    torch==2.6.0 numpy==1.26.4 transformers==4.51.3 huggingface_hub==0.30.2 \
    "ray[default]==2.43.0" vllm==0.8.4 -i $TENCENT
fi
if [ ! -d /root/ebgrpo/verl ]; then
  cd /root/ebgrpo
  git clone --depth 1 --branch v0.8.0 https://github.com/verl-project/verl.git
fi
cd /root/ebgrpo/verl
pip install --no-deps -e . -i $TENCENT
pip install $PIP_TIMEOUT \
  accelerate==1.6.0 codetiming datasets==3.5.0 dill hydra-core==1.3.2 \
  "pandas==2.2.3" "pyarrow==19.0.0" peft==0.15.1 pylatexenc \
  "tensordict==0.8.0" "torchdata==0.10.0" wandb packaging \
  uvicorn fastapi latex2sympy2_extended math_verify tensorboard \
  -i $TENCENT
pip freeze > /root/ebgrpo/freeze_20261024.txt
python -c "import vllm, torch, transformers, ray; print('SMOKE-IMPORT-OK', torch.__version__, torch.cuda.is_available())"
echo SETUP_DONE
