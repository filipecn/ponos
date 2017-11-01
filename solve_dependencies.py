#!/usr/bin/python

import os
import shutil
from subprocess import call
import platform
import urllib.request
import sys

msversion = "Visual Studio 15 2017 Win64"

root_path = os.getcwd() + "/external"
if os.path.exists(root_path + '/include'):
    shutil.rmtree(root_path + '/include')
if os.path.exists(root_path + '/lib'):
    shutil.rmtree(root_path + '/lib')
os.mkdir(root_path + '/include')
os.mkdir(root_path + '/lib')
if platform.system() != 'Windows':
    # TRIANGLE #################################################################
    os.chdir(root_path)
    build_path = root_path + '/triangle_build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.mkdir(build_path)
    os.chdir(build_path)
    call(['cmake'] + ['../triangle'])
    call(['make'])
    os.chdir(root_path)
    shutil.move(build_path + '/libtriangle.a', 'lib')
    shutil.copyfile('triangle/triangle.h', 'include/triangle.h')
    shutil.rmtree('triangle_build')
# TINYOBJ ######################################################################
os.chdir(root_path)
build_path = root_path + '/tinyobj_build'
if os.path.exists(build_path):
    shutil.rmtree(build_path)
os.mkdir(build_path)
os.chdir(build_path)
if platform.system() != 'Windows':
    call(['cmake'] + ['../tinyobj'])
    call(['make'])
    os.chdir(root_path)
    shutil.move(build_path + '/libtinyobjloader.a', 'lib')
    shutil.copyfile('tinyobj/tiny_obj_loader.h', 'include/tiny_obj_loader.h')
    shutil.rmtree('tinyobj_build')
else:
    call(['cmake'] + ['../tinyobj'] + ['-G'] + [msversion])

if 'glfw3' in sys.argv or 'all' in sys.argv:
    # GLFW3 ####################################################################
    os.chdir(root_path)
    call(['git'] + ['clone'] + ['https://github.com/glfw/glfw'])
    os.chdir(root_path + '/glfw')
    if platform.system() != 'Windows':
        call(['cmake'] + ['-G'] + ['Unix Makefiles'])
        call(['make'])
        shutil.move(root_path + '/glfw/src/libglfw3.a', root_path + '/lib')
    else:
        call(['cmake'] + ['-G'] + [msversion])
        call(
            [r"MSBuild.exe"] + [r"/p:Configuration=Release"] + ["glfw.sln"],
            shell=True)
        shutil.move(root_path + '/glfw/src/Release/glfw3.lib',
                    root_path + '/lib')
    shutil.copytree(root_path + '/glfw/include/GLFW',
                    root_path + '/include/GLFW')
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
