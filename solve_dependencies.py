#!/usr/bin/python

import os
import shutil
from subprocess import call

root_path = os.getcwd() + "/external"
if os.path.exists(root_path + '/include'):
    shutil.rmtree(root_path + '/include')
if os.path.exists(root_path + '/lib'):
    shutil.rmtree(root_path + '/lib')
os.mkdir(root_path + '/include')
os.mkdir(root_path + '/lib')

# TRIANGLE
os.chdir(root_path)
build_path = root_path + '/triangle_build'
os.mkdir(build_path)
os.chdir(build_path)
call(['cmake'] + ['../triangle'])
call(['make'])
os.chdir(root_path)
shutil.move(build_path + '/libtriangle.a', 'lib')
shutil.copyfile('triangle/triangle.h', 'include/triangle.h')
shutil.rmtree('triangle_build')

# TINYOBJ
os.chdir(root_path)
build_path = root_path + '/tinyobj_build'
os.mkdir(build_path)
os.chdir(build_path)
call(['cmake'] + ['../tinyobj'])
call(['make'])
os.chdir(root_path)
shutil.move(build_path + '/libtinyobjloader.a', 'lib')
shutil.copyfile('tinyobj/tiny_obj_loader.h', 'include/tiny_obj_loader.h')
shutil.rmtree('tinyobj_build')


# FREETYPE2
os.chdir(root_path)
call(['wget'] + ['http://download.savannah.gnu.org/releases/freetype/' +
                 'freetype-2.8.tar.gz'])
call(['tar'] + ['-vxf'] + ['freetype-2.8.tar.gz'])
os.chdir('freetype-2.8')
build_path = root_path + '/freetype-2.8/build'
os.mkdir(build_path)
os.chdir(build_path)
call(['cmake'] + ['..'])
call(['make'])
shutil.move(root_path + '/freetype-2.8/include/freetype',
            root_path + '/include')
shutil.move(root_path + '/freetype-2.8/include/ft2build.h',
            root_path + '/include')
shutil.move(build_path + '/libfreetype.a', root_path + '/lib')
shutil.rmtree(root_path + '/freetype-2.8', ignore_errors=True)
os.remove(root_path + '/freetype-2.8.tar.gz')

# LIBPNG
os.chdir(root_path)
call(['wget'] + ['http://prdownloads.sourceforge.net/libpng/' +
                 'libpng-1.6.29.tar.gz'])
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

# ZLIB
os.chdir(root_path)
call(['wget'] + ['https://zlib.net/zlib-1.2.11.tar.gz'])
call(['tar'] + ['-vxf'] + ['zlib-1.2.11.tar.gz'])
os.chdir(root_path + '/zlib-1.2.11')
build_path = root_path + '/zlib-1.2.11/build'
os.mkdir(build_path)
os.chdir(build_path)
call(['cmake'] + ['..'])
call(['make'])
shutil.move(build_path + '/libz.a', root_path + '/lib')
shutil.rmtree(root_path + '/zlib-1.2.11', ignore_errors=True)
os.remove(root_path + '/zlib-1.2.11.tar.gz')

# GLFW3
os.chdir(root_path)
call(['wget'] + ['https://github.com/glfw/glfw/releases/download/3.2.1/' +
                 'glfw-3.2.1.zip'])
call(['unzip'] + ['glfw-3.2.1.zip'])
os.chdir(root_path + '/glfw-3.2.1')
build_path = root_path + '/glfw-3.2.1/build'
os.mkdir(build_path)
os.chdir(build_path)
call(['cmake'] + ['-G'] + ['Unix Makefiles'] + ['..'])
call(['make'])
shutil.move(build_path + '/src/libglfw3.a', root_path + '/lib')
shutil.move(root_path + '/glfw-3.2.1/include/GLFW', root_path + '/include')
shutil.rmtree(root_path + '/glfw-3.2.1', ignore_errors=True)
os.remove(root_path + '/glfw-3.2.1.zip')
