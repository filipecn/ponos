#!/usr/bin/python
import os
import shutil
from subprocess import call

cur_path = os.getcwd()
build_path = os.getcwd() + "/build"

if os.path.exists(build_path):
    shutil.rmtree(build_path)

os.mkdir(build_path)
os.chdir(build_path)
call(["cmake", cur_path])
make_result = call(["make"])

# run tests
if make_result == 0 :
    tests = filter(lambda x: x.find('test', 0) == 0, os.listdir(cur_path + "/tests"))
    os.chdir(build_path + "/tests")
    call(["./run_tests"] + tests)
