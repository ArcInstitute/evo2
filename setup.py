import os
import subprocess
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from distutils.dir_util import mkpath

class VortexInstallCommand(Command):
    description = 'Install vortex submodule'

    def initialize_options(self):
        """Set default values for options"""
        pass
    
    def finalize_options(self):
        """Post-process options"""
        pass
    
    def run(self):
        vortex_dir = os.path.join(os.path.dirname(__file__), 'vortex')
        original_dir = os.getcwd()
        os.chdir(vortex_dir)
        subprocess.check_call(['make', 'setup-full'])
        os.chdir(original_dir)


class CustomInstall(install):
    def run(self):
        # Create necessary directories
        if self.build_lib:
            mkpath(os.path.join(self.build_lib, 'evo2'))
        if hasattr(self, 'build_scripts'):
            mkpath(self.build_scripts)
        install.run(self)
        self.run_command('install_vortex')


import os
import subprocess
import sys
from setuptools import setup, find_packages, Command
from setuptools.command.install import install

class VortexInstallCommand(Command):
    description = 'Install vortex submodule'

    def initialize_options(self):
        """Set default values for options"""
        pass
    
    def finalize_options(self):
        """Post-process options"""
        pass
    
    def run(self):
        vortex_dir = os.path.join(os.path.dirname(__file__), 'vortex')
        original_dir = os.getcwd()
        try:
            os.chdir(vortex_dir)
            subprocess.check_call(['make', 'setup-full'])
        finally:
            os.chdir(original_dir)

class CustomInstall(install):
    def run(self):
        install.run(self)
        self.run_command('install_vortex')

def parse_requirements(filename):
    requirements = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

requirements = parse_requirements("requirements.txt")

setup(
    name='evo2',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    cmdclass={
        'install': CustomInstall,
        'install_vortex': VortexInstallCommand,
    },
    package_dir={
        'vortex': 'vortex/vortex',
    },
    include_package_data=True,
    python_requires='>=3.1',
    license="Apache-2.0",
    description='Evo 2 project package',
    author='Evo 2 team',
    url='https://github.com/arcinstitute/evo2',
)