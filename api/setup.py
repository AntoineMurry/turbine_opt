from setuptools import setup, find_packages

requirements = """
Flask-Cors>=3.0.6
Flask-SQLAlchemy==2.3.2
Flask>=1.0.2
PyJWT
Requests
awesome-slugify==1.6.5
click
coverage
flask-restplus>=0.11
gunicorn
numpy
pandas==0.23.4
pytest
pytest-cov
pytest-flask
waitress
xlsxwriter
xlrd
sendgrid
pyarrow
google-cloud==0.28.0
google-cloud-speech==0.30.0
proto-google-cloud-datastore-v1==0.90.4
proto-google-cloud-error-reporting-v1beta1==0.15.3
proto-google-cloud-logging-v2==0.91.3
google-cloud-translate==1.3.1
google-cloud-runtimeconfig==0.28.1
google-cloud-resource-manager==0.28.1
protobuf==3.6.1
"""
# numpy==1.14.3

setup(
    name='turbine_api',
    setup_requires=['setuptools_scm'],
    use_scm_version={"root": ".."},
    author='Murry',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    scripts=['scripts/turbine_opt_api_run'],
    # all functions @cli.command() decoreated in turbine_api/cli.py
    # will be scripts for Flask API
    entry_points={'console_scripts':
                  ['turbine_opt_train_apicli = turbine_opt_train_api.cli:cli']}
)
