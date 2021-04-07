import os
import setuptools
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class get_pybind_include(object):
    """ Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


ext_modules = [
    Extension(
        'fast_norbert_cpp',
        # sort input source files to ensure bit-for-bit reproducible builds
        sorted([
            'src/fast_norbert_cpp/norbert.cxx',
            ]),
        include_dirs=[
            # path to pybind11 headers
            get_pybind_include(),
        ],
        language='c++'
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """ Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cxx', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """ Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))

            extra_compile_args=[
                            '-Wno-parentheses',
                            '-Werror',
                            # '-fopenmp',
                            '-ffast-math',
                            '-march=native',
                            ]

            opts.extend(extra_compile_args)

            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


# Get the long description from the README file
with open('README.md') as f:
    long_description = f.read()

# from src.fast_norbert import __version__
with open('src/fast_norbert/__init__.py') as f:
    s = list(f)[0]
    version = s.split()[-1][1:-1]

setup(
    name='fast_norbert',
    version=version,
    author='Artyom Palvelev',
    url='https://github.com/artyompal/fast_norbert',
    description='Accelerated Wiener filter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={"": "src"},
    packages=['fast_norbert'],
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['numpy>=1.18.1', 'scipy>=1.5.1'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
