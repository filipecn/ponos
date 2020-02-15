#!/usr/bin/py

import shutil
from pathlib import Path
import errno
import os


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


if __name__ == "__main__":
    ponos_dir = os.getcwd()
    output_dir = ponos_dir + '/build/coverage'
    shutil.rmtree(output_dir)
    Path(cwd + '/build/coverage').mkdir(parents=True, exist_ok=True)
    # copy('/mnt/windows/Projects/ponos/build/')
