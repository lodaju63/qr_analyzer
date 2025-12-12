# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

# Fix RecursionError: increase recursion limit
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

# Get ultralytics data files
ultralytics_path = None
try:
    import ultralytics
    ultralytics_path = Path(ultralytics.__file__).parent
except:
    pass

# Get dynamsoft data files
dynamsoft_path = None
try:
    import dynamsoft_barcode_reader_bundle
    dynamsoft_path = Path(dynamsoft_barcode_reader_bundle.__file__).parent
except:
    pass

# Collect datas
datas = []
if ultralytics_path:
    datas.append((str(ultralytics_path / 'cfg'), 'ultralytics/cfg'))
    datas.append((str(ultralytics_path / 'models'), 'ultralytics/models'))

if dynamsoft_path:
    # Dynamsoft dylib files (Mac)
    for dylib_file in dynamsoft_path.glob('*.dylib'):
        datas.append((str(dylib_file), 'dynamsoft_barcode_reader_bundle'))
    
    # Dynamsoft SO files (Mac)
    for so_file in dynamsoft_path.glob('*.so'):
        datas.append((str(so_file), 'dynamsoft_barcode_reader_bundle'))
    
    # Dynamsoft license files
    for lic_file in dynamsoft_path.glob('*.lic'):
        datas.append((str(lic_file), 'dynamsoft_barcode_reader_bundle'))
    
    # Dynamsoft Models folder
    models_dir = dynamsoft_path / 'Models'
    if models_dir.exists():
        datas.append((str(models_dir), 'dynamsoft_barcode_reader_bundle/Models'))
    
    # Dynamsoft Templates folder
    templates_dir = dynamsoft_path / 'Templates'
    if templates_dir.exists():
        datas.append((str(templates_dir), 'dynamsoft_barcode_reader_bundle/Templates'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'pyqtgraph',
        'numpy',
        'cv2',
        'PIL',
        'PIL.Image',
        'dynamsoft_barcode_reader_bundle',
        'dynamsoft_barcode_reader_bundle.cvr',
        'dynamsoft_barcode_reader_bundle.dbr',
        'dynamsoft_barcode_reader_bundle.core',
        'dynamsoft_barcode_reader_bundle.dip',
        'dynamsoft_barcode_reader_bundle.dnn',
        'dynamsoft_barcode_reader_bundle.license',
        'dynamsoft_barcode_reader_bundle.utility',
        'ultralytics',
        'ultralytics.nn',
        'ultralytics.nn.modules',
        'ultralytics.nn.tasks',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.engine',
        'ultralytics.engine.predictor',
        'ultralytics.utils',
        'ultralytics.utils.checks',
        'ultralytics.utils.downloads',
        'ultralytics.data',
        'torch',
        'torch._C',
        'torch._dynamo',
        'torchvision',
        'torchvision.models',
        'yaml',
        'tqdm',
        'pandas',
        'scipy',
        'scipy.ndimage',
        'scipy.spatial',
        'scipy.spatial.distance',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends',
        'matplotlib.backends.backend_agg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_ultralytics.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single executable for Mac
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='QR_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console on Mac
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Create .app bundle
app = BUNDLE(
    exe,
    name='QR_Analyzer.app',
    icon=None,
    bundle_identifier='com.qranalyzer.app',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Video Files',
                'CFBundleTypeRole': 'Viewer',
                'LSItemContentTypes': ['public.movie'],
            }
        ]
    },
)
