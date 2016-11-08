#!/usr/bin/python
import os
import shutil
from subprocess import call
import sys
import glob
import subprocess
import platform

cur_path = os.getcwd()
build_path = os.getcwd() + "/build"

if len(sys.argv) < 2 :
    print('all - remove cache, make, test and generate docs')
    print('make - just make')
    print('test - make and test')
    print('docs [optional - output dir] - generate docs')
    sys.exit(1)

if sys.argv[1] == 'docs':
    cldoc_command = "cldoc generate -std=c++11 -I/home/filipecn/gitclones/OpenVDB/include -Iaergia/src -Iponos/src -Ihelios/src -Ihercules/src -Iposeidon/src -fPIC -- --report --merge docs --type html --static --output "
    outputdir = 'html'
    if len(sys.argv) > 2:
        outputdir = sys.argv[2]
    cldoc_command += outputdir
    for filename in glob.iglob('aergia/src/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('aergia/src/**/*.h*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('ponos/src/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('ponos/src/**/*.h*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('helios/src/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('helios/src/**/*.h*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('hercules/src/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('hercules/src/**/*.h*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('poseidon/src/**/*.c*', recursive=True):
        cldoc_command += " " + filename
    for filename in glob.iglob('poseidon/src/**/*.h*', recursive=True):
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
    if platform.system() == 'Windows':
        call(["cmake"] + ['-G'] + ['Visual Studio 14 2015 Win64'] + sys.argv[2:] + [cur_path])
    else:
        call(["cmake"] + sys.argv[2:] + [cur_path])

if platform.system() == 'Windows':
    make_result = call([r"C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"] + ["PONOS.sln"], shell=True)
else:
    make_result = call(["make"], shell=True)

if make_result != 0 :
    sys.exit(1)

if sys.argv[1] != 'test':
    sys.exit(0)

# run tests
test_libs = ['ponos', 'hercules']#, 'poseidon']
for l in test_libs:
    tests = list(filter(lambda x: x.find('Test', 0) == 0, os.listdir(cur_path + "/" + l + "/tests")))
    if os.path.isdir(build_path + "/" + l + "/tests") == True:
        os.chdir(build_path + "/" + l + "/tests")
        print(["./run_" + l + "_tests"] +[x[:-4] for x in tests])
        if platform.system() == 'Windows':
            result = call(["./Debug/run_" + l + "_tests.exe"] +[x[:-4] for x in tests])
        else:
            result = call(["./run_" + l + "_tests"] +[x[:-4] for x in tests])

sys.exit(result)
