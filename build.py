#!/usr/bin/python
import os
import shutil
from subprocess import call
import sys
import platform

mingw_make = 'C:/MinGW/bin/mingw32-make.exe'
msversion = "Unix Makefiles"
if platform.system() == 'Windows' or 'mingw' in sys.argv:
    msversion = "MinGW Makefiles"
if 'msvc' in sys.argv:
    msversion = "Visual Studio 15 2017 Win64"

src_path = os.getcwd()
build_path = os.getcwd() + "/build"
if platform.system() == 'Windows':
    build_path = os.getcwd() + "/wbuild"

if 'test' in sys.argv:
    os.chdir(build_path)
    r = call(["make tests"], shell=True)
    exit(r)

if 'docs' in sys.argv:
    call(["doxygen Doxyfile"], shell=True)
    call(
        ["xsltproc doc/xml/combine.xslt doc/xml/index.xml > doc/xml/all.xml"],
        shell=True)
    call(["python doxml2md.py doc/xml/all.xml"], shell=True)
    sys.exit(0)

if 'all' in sys.argv or not os.path.exists(build_path):
    #if os.path.exists(build_path):
    #    shutil.rmtree(build_path, ignore_errors=True)
    if not os.path.exists(build_path):
        os.mkdir(build_path)
    os.chdir(build_path)
    call(["cmake"] + ['-G'] + [msversion] + sys.argv[2:] + [src_path])

os.chdir(build_path)

if platform.system() == 'Windows':
    if 'MinGW Makefiles' == msversion:
        make_result = call([mingw_make], shell=True)
        if "-DTRAVIS=1" not in sys.argv:
            call([mingw_make] + ["install"], shell=True)
    else:
        make_result = call([r"MSBuild.exe"] + [r"/p:Configuration=Release"] + [r"/p:Machine=X64"] + ["PONOS.sln"], shell=True)
else:
    make_result = call(["make -j8"], shell=True)
    if "-DTRAVIS=1" not in sys.argv:
        call(["make install"], shell=True)

if make_result != 0:
    sys.exit(1)
