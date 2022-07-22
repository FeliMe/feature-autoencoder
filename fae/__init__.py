import os

DATAROOT = os.environ.get('DATAROOT')
CAMCANROOT = os.path.join(DATAROOT, 'CamCAN')
BRATSROOT = os.path.join(DATAROOT, 'BraTS')
MOODROOT = os.path.join(DATAROOT, 'MOOD')
PHYSIONETROOT = os.path.join(DATAROOT, 'Physionet-ICH')

WANDBNAME = os.environ.get('WANDBNAME')
WANDBPROJECT = os.environ.get('WANDBPROJECT')
WANDBDIR = os.environ.get(
    'WANDBDIR',
    os.path.join(os.path.expanduser('~'), 'wandb')
)
