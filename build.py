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
    call(
        ["xsltproc doc/xml/combine.xslt doc/xml/index.xml > doc/xml/all.xml"],
        shell=True)
    call(["python doxml2md.py doc/xml/all.xml"], shell=True)
    sys.exit(1)

if os.path.exists(build_path) and sys.argv[1] == 'all':
    shutil.rmtree(build_path, ignore_errors=True)

if not os.path.exists(build_path):
    os.mkdir(build_path)

os.chdir(build_path)

if sys.argv[1] == 'all':
    if platform.system() == 'Windows':
        call(["cmake"] + ['-G'] + ['Visual Studio 15 2017 Win64'] +
             sys.argv[2:] + [cur_path])
    else:
        call(["cmake"] + sys.argv[2:] + [cur_path])

if platform.system() == 'Windows':
    make_result =\
        call([r"MSBuild.exe"] + [r"/p:Configuration=Release"] + ["PONOS.sln"],
             shell=True)
else:
    make_result = call(["make -j8"], shell=True)

if make_result != 0:
    sys.exit(1)

# install
include_path = build_path + '/include'
lib_path = build_path + '/lib'
if os.path.exists(include_path):
    shutil.rmtree(include_path, ignore_errors=True)
if os.path.exists(lib_path):
    shutil.rmtree(lib_path, ignore_errors=True)
if not os.path.exists(include_path):
    os.mkdir(include_path)
if not os.path.exists(lib_path):
    os.mkdir(lib_path)

# , 'poseidon', 'helios']
libs = ['ponos', 'aergia']
for l in libs:
    if not os.path.exists(build_path + '/' + l):
        continue
    if not os.path.exists(include_path + '/' + l):
        os.mkdir(include_path + '/' + l)
    for root, subdirs, files in os.walk(cur_path + '/' + l):
        path = root.split(l + '/src/')
        if platform.system() == 'Windows':
            path = root.split(l + '\\src\\')
        if len(path) > 1:
            dst = include_path + '/' + l + '/' + path[1]
            os.makedirs(dst)
            for f in files:
                if '.h' in f or '.inl' in f:
                    shutil.copyfile(root + '/' + f, dst + '/' + f)
                    print('copying ' + root + '/' + f + ' to ' + dst + '/' + f)
    shutil.copyfile(cur_path + '/' + l + '/src/' + l + '.h',
                    include_path + '/' + l + '/' + l + '.h')
    if platform.system() != 'Windows':
        shutil.copyfile(build_path + '/' + l + '/lib' + l + '.a',
                        lib_path + '/lib' + l + '.a')
    else:
        shutil.copyfile(build_path + '/' + l + '/Release/' + l + '.lib',
                        lib_path  + l + '.lib')
