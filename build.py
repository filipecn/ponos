#!/usr/bin/python
import os
import shutil
from subprocess import call
import sys

cur_path = os.getcwd()
build_path = os.getcwd() + "/build"

if os.path.exists(build_path):
    shutil.rmtree(build_path)

os.mkdir(build_path)
os.chdir(build_path)
print 'Argument List:', str(sys.argv[1:])
call(["cmake"] + sys.argv + [cur_path])
make_result = call(["make"])

if make_result != 0 :
    sys.exit(1)

# run tests
tests = filter(lambda x: x.find('Test', 0) == 0, os.listdir(cur_path + "/tests"))
os.chdir(build_path + "/tests")
result = call(["./run_tests"] + map(lambda test : test[:-4], tests))

sys.exit(result)
