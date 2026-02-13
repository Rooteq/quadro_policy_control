from setuptools import find_packages, setup
import os
import glob

package_name = 'quadro_policy_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'policy'), glob.glob(os.path.join('policy', '*.pt')) + glob.glob(os.path.join('policy', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rooteq',
    maintainer_email='szymerut@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'quadro_controller = quadro_policy_control.quadro_controller:main'
        ],
    },
)
