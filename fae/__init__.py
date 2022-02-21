import os

DATAROOT = os.environ.get('DATAROOT')
CAMCANROOT = os.path.join(DATAROOT, 'CamCAN')
BRATSROOT = os.path.join(DATAROOT, 'BraTS')

WANDBNAME = os.environ.get('WANDBNAME')
WANDBPROJECT = os.environ.get('WANDBPROJECT')
