import os
import shutil
import subprocess

APPNAME_TO_DIRECTORY = {
    'correct-segmentation': 'correct-segmentation',
    'measure-length': 'measure-length',
    'remove-background': 'remove-background',
    'segmentation': 'segmentation',
    'stitch-images': 'stitch-images'
}

files = ['requirements.txt', 'Procfile', 'DOKKU_SCALE']
for f in files:
    shutil.copyfile(
        os.path.join(
            'apps',
            APPNAME_TO_DIRECTORY[os.environ['DASH_APP_NAME']],
            f
        ),
        f
    )

subprocess.run('python -m pip install -r requirements.txt'.split(' '))
