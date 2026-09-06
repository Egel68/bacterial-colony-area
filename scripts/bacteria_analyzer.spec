# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

root = Path(SPEC).resolve().parent.parent
os.chdir(str(root))

block_cipher = None

a = Analysis(
    [str(root / 'main.py')],
    pathex=[str(root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'albumentations',
        'tqdm',
        'onnxruntime',
        'albumentations.augmentations.transforms',
        'albumentations.augmentations.geometric.transforms',
        'albumentations.augmentations.dropout.transforms',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'tensorboard', 'plotly',
        'jinja2', 'fastapi', 'uvicorn', 'websockets', 'rich',
        'nuitka', 'zstandard', 'onnxscript', 'timm',
        'train',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BacteriaAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / 'icon.ico'),
)
