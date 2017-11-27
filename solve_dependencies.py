#!/usr/bin/python

import os
import shutil
from subprocess import call
import platform
import urllib.request
import sys

mingw_make = 'C:/MinGW/bin/mingw32-make.exe'
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

exit(0)
if 'freetype' in sys.argv or 'all' in sys.argv:
    # GRAPHITE #################################################################
    os.chdir(root_path)
    call(['git'] + ['clone'] + ['https://github.com/silnrsi/graphite'])
    os.chdir(root_path + '/graphite')
    if platform.system() != 'Windows':
        call(['cmake'] + ['-G'] + ['Unix Makefiles'])
        call(['make'])
        shutil.move(root_path + '/graphite/src/libgraphite2.so.3.0.1',
                    root_path + '/lib')

    # HARFBUZZ #################################################################
    os.chdir(root_path)
    urllib.request.urlretrieve(
        'https://github.com/behdad/harfbuzz/releases/download/1.6.3/' +
        'harfbuzz-1.6.3.tar.bz2', 'harfbuzz-1.6.3.tar.bz2')
    call(['tar'] + ['-vxf'] + ['harfbuzz-1.6.3.tar.bz2'])
    build_path = root_path + '/harfbuzz_build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)
    os.chdir(build_path)
    if platform.system() != 'Windows':
        call(['cmake'] + ['-G'] + ['Unix Makefiles'] + ['../harfbuzz-1.6.3'])
        call(['make'])
        shutil.move(build_path + '/libharfbuzz.a', root_path + '/lib')
    if platform.system() != 'Windows':
        # LIBPNG ###############################################################
        os.chdir(root_path)
        call(['wget'] + [
            'http://prdownloads.sourceforge.net/libpng/' +
            'libpng-1.6.29.tar.gz'
        ])
        call(['tar'] + ['-vxf'] + ['libpng-1.6.29.tar.gz'])
        os.chdir(root_path + '/libpng-1.6.29')
        build_path = root_path + '/libpng-1.6.29/build'
        os.mkdir(build_path)
        os.chdir(build_path)
        call(['cmake'] + ['..'])
        call(['make'])
        shutil.move(build_path + '/libpng16.a', root_path + '/lib')
        shutil.rmtree(root_path + '/libpng-1.6.29', ignore_errors=True)
        os.remove(root_path + '/libpng-1.6.29.tar.gz')
    # BZ2LIB ###################################################################
    os.chdir(root_path)
    urllib.request.urlretrieve('http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz',
                               'bzip2-1.0.6.tar.gz')
    call(['tar'] + ['-vxf'] + ['bzip2-1.0.6.tar.gz'])
    folder = 'bzip2-1.0.6'
    os.chdir('/'.join([root_path, folder]))
    if platform.system() != 'Windows':
        call(['make'])
        shutil.move('/'.join([root_path, folder, 'libbz2.a']),
                    root_path + '/lib')
    shutil.rmtree(root_path + '/' + folder, ignore_errors=True)
    os.remove(root_path + '/bzip2-1.0.6.tar.gz')
    # ZLIB #####################################################################
    os.chdir(root_path)
    urllib.request.urlretrieve('https://zlib.net/zlib-1.2.11.tar.gz',
                               'zlib-1.2.11.tar.gz')
    call(['tar'] + ['-vxf'] + ['zlib-1.2.11.tar.gz'])
    os.chdir(root_path + '/zlib-1.2.11')
    build_path = root_path + '/zlib-1.2.11/build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)
    os.chdir(build_path)
    if platform.system() != 'Windows':
        call(['cmake'] + ['..'])
        call(['make'])
        shutil.move(build_path + '/libz.a', root_path + '/lib')
    else:
        call(['cmake'] + ['..'] + ['-G'] + [msversion])
        call(
            [r"MSBuild.exe"] + [r"/p:Configuration=Release"] + ["zlib.sln"],
            shell=True)
        shutil.move(build_path + '/Release/zlib.lib', root_path + '/lib')
    shutil.rmtree(root_path + '/zlib-1.2.11', ignore_errors=True)
    os.remove(root_path + '/zlib-1.2.11.tar.gz')

    # FREETYPE2 ################################################################
    os.chdir(root_path)
    urllib.request.urlretrieve(
        'http://download.savannah.gnu.org/releases/freetype/' +
        'freetype-2.8.1.tar.gz', 'freetype-2.8.1.tar.gz')
    call(['tar'] + ['-vxf'] + ['freetype-2.8.1.tar.gz'])
    os.chdir('freetype-2.8.1')
    build_path = root_path + '/freetype-2.8.1/build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)
    os.chdir(build_path)
    if platform.system() != 'Windows':
        call(['cmake'] + ['..'])
        call(['make'])
        shutil.move(build_path + '/libfreetype.a', root_path + '/lib')
    else:
        call(['cmake'] + ['..'] + ['-G'] + [msversion])
        call(
            [r"MSBuild.exe"] + [r"/p:Configuration=Release"] +
            ["freetype.sln"],
            shell=True)
        shutil.move(build_path + '/Release/freetype.lib', root_path + '/lib')
    shutil.move(root_path + '/freetype-2.8.1/include/freetype',
                root_path + '/include')
    shutil.move(root_path + '/freetype-2.8.1/include/ft2build.h',
                root_path + '/include')
    shutil.rmtree(root_path + '/freetype-2.8.1', ignore_errors=True)
    os.remove(root_path + '/freetype-2.8.1.tar.gz')
