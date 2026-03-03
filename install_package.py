#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import os
# import subprocess
# import sys

# py_version = sys.version
# print(py_version)
# if py_version[:4] == '3.9' or py_version[:4] == '3.10' or py_version[:4] == '3.11':
#     py_requires = 'python' + sys.version[:4]
# else:
#     py_requires = 'python3.8'
# print(py_requires)

# file = os.getcwd() 


# #subprocess.run(["cd",file]), check=True, stdout=subprocess.PIPE).stdout
# os.system('cd ' + file)

# os.system('./make_version.sh')

# print("version file updated")
# print('*'*100)

# # subprocess.run(["git", "add", "."], check=True, stdout=subprocess.PIPE).stdout
# # subprocess.run(["git", "commit", "-am", "Bug fix commit"], check=True, stdout=subprocess.PIPE).stdout

# # print('git commit done')

# subprocess.run(["git", "pull"], check=True, stdout=subprocess.PIPE).stdout
# print('git pull done')
# print('*'*100)

# subprocess.run(["git", "push"], check=True, stdout=subprocess.PIPE).stdout
# print('*'*100)
# print('removing dist and build folders')

# if os.path.exists(file+'/dist'):
#     os.system('sudo rm -rf '+file+'/dist')
#     os.system('sudo rm -rf '+file+'/build')
# #subprocess.run(["ls"]),check=True, stdout=subprocess.PIPE).stdout
# os.system("ls")

# os.system(py_requires + ' -m build')

# print('*'*100)
# print('wheel built')
# print(py_requires + ' -m pip install '+file + '/dist/' +os.listdir(file +'/dist')[-1] + ' --break-system-packages')
# os.system(py_requires + ' -m pip install '+file + '/dist/' +os.listdir(file +'/dist')[-1] + ' --break-system-packages')

# # print('package installed')
# # print('*'*100)
# # os.system(py_requires + ' -m twine upload dist/*')

import os
import subprocess
import glob
import sys

def install_package():
    subprocess.run(['./make_version.sh'], check=True)
    subprocess.run(['git', 'add', '.'], check=True)
    msg = "updated version :" + subprocess.run(['cat', 'VERSION.txt'], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
    # skip commit if make_version.sh already committed everything
    status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(['git', 'commit', '-am', msg], check=True)
        print('git commit done')
    else:
        print('nothing to commit, skipping')
    subprocess.run(['git', 'pull'], check=True)
    print('git pull done')
    subprocess.run(['git', 'push'], check=True)
    # subprocess.run(['git', 'push', '--tags'], check=True)
    version = subprocess.run(
    ['cat', 'VERSION.txt'],
    capture_output=True,
    text=True).stdout.strip()

    # Create tag locally if it doesn't exist
    subprocess.run(['git', 'tag', version], check=False)

    # Push ONLY this tag
    subprocess.run(['git', 'push', 'origin', version], check=False)

    print('git push done')
    print('*'*100)
    subprocess.run(['rm', '-rf', 'dist'], check=True)
    subprocess.run(['uv', 'build'], check=True)
    latest_file = sorted(os.listdir('./dist'))[-1]
    whl_files = [f for f in os.listdir('./dist') if f.endswith('.whl')]
    latest_file = sorted(whl_files)[-1] if whl_files else sorted(os.listdir('./dist'))[-1]
    print(f'Installing package file: {latest_file}')
    prefix = os.environ.get('MB_UV_PREFIX', os.path.expanduser('~/.local'))
    print(f'Installing into prefix: {prefix}')
    subprocess.run(
        [
            'uv',
            'pip',
            'install',
            f'./dist/{latest_file}',
            '--python',
            sys.executable,
            '--prefix',
            prefix,
        ],
        check=True,
    )
        
install_package()
print('package installed')
print('*'*100)

whl_files = glob.glob("dist/*.whl")
subprocess.run(["uvx", "uv-publish"] + whl_files, check=True)

# Upload .whl to the latest GitHub release
version = subprocess.run(['cat', 'VERSION.txt'], capture_output=True, text=True).stdout.strip()
# Create the release if it doesn't exist yet
subprocess.run(['gh', 'release', 'create', version, '--title', f'v{version}', '--notes', f'Release {version}', '--latest'], check=False)
for whl in whl_files:
    subprocess.run(['gh', 'release', 'upload', version, whl, '--clobber'], check=True)
print(f'Uploaded {len(whl_files)} whl file(s) to GitHub release {version}')
