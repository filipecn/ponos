#!/usr/bin/python
import os
import shutil
from subprocess import call
import sys
import platform

cur_path = os.getcwd()
build_path = os.getcwd() + "/build"
if platform.system() == 'Windows':
    build_path = os.getcwd() + "/win_build"

if len(sys.argv) < 2:
    print('all - remove cache, make, test and generate docs')
    print('make - just make')
    print('test - make and test')
    print('docs [optional - output dir] - generate docs')
    sys.exit(1)

if sys.argv[1] == 'docs':
    call(["doxygen Doxyfile"], shell=True)
    call(["xsltproc doc/xml/combine.xslt doc/xml/index.xml > doc/xml/all.xml"],
         shell=True)
    call(["python doxml2md.py doc/xml/all.xml"], shell=True)
    sys.exit(1)

if os.path.exists(build_path) and sys.argv[1] == 'all':
    shutil.rmtree(build_path)

first = False
if not os.path.exists(build_path):
    os.mkdir(build_path)
    first = True

os.chdir(build_path)

if first:
    if platform.system() == 'Windows':
        call(["cmake"] + ['-G'] + ['Visual Studio 14 2015 Win64'] +
             sys.argv[2:] + [cur_path])
    else:
        call(["cmake"] + sys.argv[2:] + [cur_path])

if platform.system() == 'Windows':
    make_result =\
        call([r"C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"] +
             [r"/p:Configuration=Release"] + ["PONOS.sln"], shell=True)
# make_result = call([r"C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"] +
#                   ["PONOS.sln"], shell=True)
else:
    make_result = call(["make -j8"], shell=True)

if make_result != 0:
    sys.exit(1)

# install
include_path = build_path + '/include'
lib_path = build_path + '/lib'
if os.path.exists(include_path):
    shutil.rmtree(include_path)
if os.path.exists(lib_path):
    shutil.rmtree(lib_path)
os.mkdir(lib_path)
os.mkdir(include_path)

# , 'poseidon', 'helios']
libs = ['ponos', 'aergia']
for l in libs:
    for root, subdirs, files in os.walk(cur_path + '/' + l):
        print(root)
        path = root.split(l + '/src/')
        if len(path) > 1:
            dst = include_path + '/' + l + '/' + path[1]
            os.makedirs(dst)
            for f in files:
                if '.h' in f or '.inl' in f:
                    shutil.copyfile(root + '/' + f, dst + '/' + f)
                    print('copying ' + root + '/' + f + ' to ' + dst + '/' + f)
    shutil.copyfile(cur_path + '/' + l + '/src/' + l + '.h',
                    include_path + '/' + l + '/' + l + '.h')
    shutil.copyfile(build_path + '/' + l + '/lib' + l + '.a',
                    lib_path + '/lib' + l + '.a')

if sys.argv[1] != 'test':
    sys.exit(0)

# run tests
result = 0
test_libs = ['ponos', 'hercules', 'poseidon']
for l in test_libs:
    tests = list(filter(lambda x: x.find('Test', 0) == 0,
                        os.listdir(cur_path + "/" + l + "/tests")))
    if os.path.isdir(build_path + "/" + l + "/tests") == True:
        os.chdir(build_path + "/" + l + "/tests")
        print(["./run_" + l + "_tests"] + [x[:-4] for x in tests])
        if platform.system() == 'Windows':
            result = call(["./Debug/run_" + l + "_tests.exe"] +
                          [x[:-4] for x in tests])
        else:
            result = call(["./run_" + l + "_tests"] + [x[:-4] for x in tests])

sys.exit(result)
