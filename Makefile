clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__
	@rm -fr __pycache__
	@rm -fr build
	@rm -fr dist
	@rm -fr turbine-opt-*.dist-info
	@rm -fr turbine-opt.egg-info

all: clean install test check_code

# install:
# 	@pip install . -U

install: clean wheel
	@pip3 install -U dist/*.whl

prod_install: wheel
	@ansible-playbook -i ansible/all.serverlist \
						--extra-vars "venv=venv" \
						--extra-vars "user=fox" \
						ansible/playbook_deploy.yml
	@echo Package installed on host remote host.


install_requirements:
	@pip install -r requirements.txt

get_dependencies:
	@python scripts/get_last_artifacts.py $(PRIVATE_TOKEN) infra/zf_tools
	@echo downloading dependencies wheel...

wheel:
	@rm -f dist/*.whl
	@python setup.py bdist_wheel  # --universal if you are python2&3

check_code:
	@flake8 scripts/* twt_maps/*.py

test:
	@coverage run -m unittest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*,turbine-opt/*
ftest:
	@Write me

uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
