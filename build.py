#!/usr/bin/python
import os
import shutil
from subprocess import call
import sys
import glob
import subprocess

cur_path = os.getcwd()
build_path = os.getcwd() + "/build"

if len(sys.argv) < 2 :
    print('all - remove cache, make, test and generate docs')
    print('make - just make')
    print('test - make and test')
    print('docs [optional - output dir] - generate docs')
    sys.exit(1)

if sys.argv[1] == 'docs':
    cldoc_command = "cldoc generate -std=c++11 -Iponos/src -Ihelios/src -Ihercules/src -fPIC -- --report --merge docs --type html --static --output "
    outputdir = 'html'
    if len(sys.argv) > 2:
        outputdir = sys.argv[2]
    cldoc_command += outputdir
    for filename in glob.iglob('ponos/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('ponos/**/*.h*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('helios/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('helios/**/*.h*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('hercules/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('hercules/**/*.h*', recursive=True):
        cldoc_command += " " + filename
    print(cldoc_command)
    call([cldoc_command], shell=True)
    if sys.argv[1] == 'docs':
        sys.exit(1)

if os.path.exists(build_path) and sys.argv[1] == 'all':
    shutil.rmtree(build_path)

first = False
if not os.path.exists(build_path):
    os.mkdir(build_path)
    first = True

os.chdir(build_path)

if first :
    call(["cmake"] + sys.argv[2:] + [cur_path])

make_result = call(["make"])

if make_result != 0 :
    sys.exit(1)

if sys.argv[1] != 'test':
    sys.exit(0)

# run tests
tests = list(filter(lambda x: x.find('Test', 0) == 0, os.listdir(cur_path + "/ponos/tests")))
print(tests)
os.chdir(build_path + "/ponos/tests")
result = call(["./run_tests"] +[x[:-4] for x in tests])

sys.exit(result)
