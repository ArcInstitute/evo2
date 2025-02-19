import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

def run_make_setup_full():
    base_dir = os.path.dirname(__file__)
    vortex_dir = os.path.join(base_dir, 'vortex')
    original_dir = os.getcwd()
    
    # If we're in a git repository, update submodules
    if os.path.exists(os.path.join(base_dir, '.git')):
        print("Updating git submodules...")
        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'], cwd=base_dir)
    
    # Ensure the Makefile uses the current Python interpreter
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    print(f"Running 'make setup-full' in {vortex_dir} with PYTHON={sys.executable} ...")
    
    try:
        os.chdir(vortex_dir)
        subprocess.check_call(['make', 'setup-full'], env=env)
    finally:
        os.chdir(original_dir)

class CustomBuildPy(_build_py):
    def run(self):
        self.run_command('egg_info')
        run_make_setup_full()
        super().run()

class CustomDevelop(_develop):
    def run(self):
        run_make_setup_full()
        super().run()

class CustomBDistWheel(_bdist_wheel):
    def run(self):
        self.run_command('egg_info')
        super().run()

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
    # Include only the evo2 package (exclude the vortex submodule)
    packages=find_packages(include=["evo2", "evo2.*"]),
    install_requires=requirements,
    cmdclass={
        'build_py': CustomBuildPy,
        'develop': CustomDevelop,
        'bdist_wheel': CustomBDistWheel,
    },
    include_package_data=True,
    python_requires='>=3.1',
    license="Apache-2.0",
    description='Evo 2 project package',
    author='Evo 2 team',
    url='https://github.com/garykbrixi/evo2',
)
