import setuptools

setuptools.setup(
    name='sagod',
    version='0.0.1',
    description='SAGOD : Static Attributed Graph Outlier Detection',
    author="Welt Xing",
    author_email="xingcy@smail.nju.edu.cn",
    maintainer="Welt Xing",
    maintainer_email="xingcy@smail.nju.edu.cn",
    packages=['sagod', 'sagod/models'],
    license='MIT License',
    install_requires=['numpy', 'sklearn', 'pyod', 'torch', 'torch_geometric'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Kaslanarian/SAGOD',
)