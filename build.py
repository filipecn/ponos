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
    print('docs - generate docs')
    sys.exit(1)

if sys.argv[1] == 'docs' or sys.argv[1] == 'all':
    cldoc_command = "cldoc generate -std=c++11 -Iponos/src -Ihelios/src -Ihercules/src -fPIC -- --report --merge docs --output html --type html --static"
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
    sys.exit(1)

# run tests
tests = filter(lambda x: x.find('Test', 0) == 0, os.listdir(cur_path + "/ponos/tests"))
os.chdir(build_path + "/tests")
result = call(["./run_tests"] + map(lambda test : test[:-4], tests))

sys.exit(result)
