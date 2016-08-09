#!/usr/bin/python
import os
import shutil
from subprocess import call
import sys

cur_path = os.getcwd()
build_path = os.getcwd() + "/build"

if len(sys.argv) < 2 :
    print 'all - remove cache, make and test'
    print 'make - just make'
    print 'test - make and test'
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
