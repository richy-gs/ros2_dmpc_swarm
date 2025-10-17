from setuptools import find_packages, setup

package_name = 'mpc_drone_viz'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roli_005',
    maintainer_email='jesusrg2405@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_controller=mpc_drone_viz.mpc_controller:main',
            'mpc_controller_ca=mpc_drone_viz.mpc_controller_ca:main',
        ],
    },
)
