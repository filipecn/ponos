#!/usr/bin/python

import os
import platform
import shutil
import sys
from subprocess import call

mingw_make = 'C:/Program Files (x86)/mingw-w64/i686-5.3.0-posix-dwarf-rt_v4-rev0/mingw32/bin/mingw32-make.exe'
msversion = "Unix Makefiles"
if platform.system() == 'Windows' or 'mingw' in sys.argv:
    msversion = "MinGW Makefiles"
if 'msvc' in sys.argv:
    msversion = "Visual Studio 15 2017 Win64"

build_root_path = os.getcwd() + '/build/external'
if platform.system() == 'Windows':
    build_root_path = os.getcwd() + '/wbuild/external'
src_root_path = os.getcwd() + "/external"

if platform.system() == 'Windows':
    if not os.path.exists(os.getcwd() + '/wbuild'):
        os.mkdir(os.getcwd() + '/wbuild')
        os.mkdir(build_root_path)
else:
    if not os.path.exists(os.getcwd() + '/build'):
        os.mkdir(os.getcwd() + '/build')
        os.mkdir(build_root_path)

if os.path.exists(build_root_path + '/include'):
    shutil.rmtree(build_root_path + '/include')
if os.path.exists(build_root_path + '/lib'):
    shutil.rmtree(build_root_path + '/lib')
os.mkdir(build_root_path + '/include')
os.mkdir(build_root_path + '/lib')

if platform.system() != 'Windows' and ('triangle' in sys.argv or 'all' in sys.argv):
    # TRIANGLE #################################################################
    src_path = src_root_path + '/triangle'
    build_path = build_root_path + '/triangle_build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)
    os.chdir(build_path)
    call(['cmake'] + [src_path] + ['-G'] + [msversion])
    call(['make'])
    shutil.copyfile(build_path + '/libtriangle.a', build_root_path + '/lib/libtriangle.a')
    lib_path = build_root_path + '/lib/libtriangle.a'
    shutil.copyfile(src_path + '/triangle.h', build_root_path + '/include/triangle.h')
    with open(build_root_path + '/dependencies.txt', 'a') as file:
        file.write('SET(TRIANGLE_LIB ' + lib_path.replace('\\', '/') + ')' + '\n')
        file.write('SET(TRIANGLE_INCLUDE ' + build_root_path.replace('\\', '/') + '/include)' + '\n')
# TINYOBJ ######################################################################
if "tinyobj" in sys.argv or 'all' in sys.argv:
    src_path = src_root_path + '/tinyobj'
    build_path = build_root_path + '/tinyobj_build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)
    os.chdir(build_path)
    call(['cmake'] + [src_path] + ['-G'] + [msversion])
    if platform.system() != 'Windows':
        call(['make'])
        shutil.move(build_path + '/libtinyobjloader.a', build_root_path + '/lib')
        lib_path = build_root_path + '/lib/libtinyobjloader.a'
        shutil.copyfile(src_path + '/tiny_obj_loader.h', build_root_path + '/include/tiny_obj_loader.h')
        with open(build_root_path + '/dependencies.txt', 'a') as file:
            file.write('SET(TINY_OBJ_LIB ' + lib_path.replace('\\', '/') + ')' + '\n')
            file.write('SET(TINY_OBJ_INCLUDE ' + build_root_path.replace('\\', '/') + '/include)' + '\n')
if 'glfw3' in sys.argv or 'all' in sys.argv:
    # GLFW3 ####################################################################
    src_path = src_root_path + '/glfw'
    build_path = build_root_path + '/glfw_build'
    if not os.path.exists(src_path):
        os.chdir(src_root_path)
        call(['git'] + ['clone'] + ['https://github.com/glfw/glfw'])
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)
    os.chdir(build_path)
    call(['cmake'] + [src_path] + ['-G'] + [msversion])  # + ['-DBUILD_SHARED_LIBS=ON'])
    if platform.system() != 'Windows':
        call(['make'] + ['-j8'])
        shutil.move(build_path + '/src/libglfw3.a', build_root_path + '/lib')
        lib_path = build_root_path + '/lib/libglfw3.a'
    else:
        if msversion == 'MinGW Makefiles':
            call([mingw_make], shell=True)
            shutil.move(build_path + '/src/libglfw3.a',
                        build_root_path + '/lib')
            lib_path = build_root_path + '/lib/libglfw3.a'
        else:
            call([r"MSBuild.exe"] + [r"/p:Configuration=Release"] + ["glfw.sln"], shell=True)
            shutil.move(build_path + '/src/Release/glfw3.lib',
                        build_root_path + '/lib')
            lib_path = build_root_path + '/lib/glfw3.lib'
    shutil.copytree(src_path + '/include/GLFW',
                    build_root_path + '/include/GLFW')
    with open(build_root_path + '/dependencies.txt', 'a') as file:
        file.write('SET(GLFW_LIB ' + lib_path.replace('\\', '/') + ')' + '\n')
        file.write('SET(GLFW_INCLUDE_DIRS ' + build_root_path.replace('\\', '/') + '/include)' + '\n')
