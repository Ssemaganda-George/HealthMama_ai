import pkg_resources

packages = ['Flask', 'Flask-Session', 'openai', 'pandas', 'sentence-transformers', 'numpy', 'scikit-learn']
for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package} == {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package} is not installed")
