$ErrorActionPreference = 'Stop'

Write-Host '[1/4] Upgrade pip tools'
python -m pip install --upgrade pip setuptools wheel

Write-Host '[2/4] Install PyTorch 2.2.2 with CUDA 12.1'
python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

Write-Host '[3/4] Install PyG extensions and PyG'
python -m pip install torch-scatter==2.1.2+pt22cu121 torch-sparse==0.6.18+pt22cu121 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
python -m pip install torch-geometric==2.5.3

Write-Host '[4/4] Install remaining dependencies'
python -m pip install -r (Join-Path $PSScriptRoot 'requirements-windows.txt')
python (Join-Path $PSScriptRoot 'verify_environment.py')
