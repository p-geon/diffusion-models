export CONTAINER_NAME = diffusion_models
export DOCKERFILE = Dockerfile
export DIR_DATA=data
export SCRIPT=ddpm_swissroll/main.py
export BUILD_QUIET=-q
#export BUILD_QUIET=``

# utils
.PHONY: build
build: ## docker build
	docker build $(BUILD_QUIET) -f $(DOCKERFILE) -t $(CONTAINER_NAME) .


# -----------
# run-variant
# -----------
void: ## run with shell
	@make build
	docker run -it --rm  \
		-v `pwd`:/work \
		-v `pwd`/$(DIR_DATA):/data \
		$(CONTAINER_NAME) \
		/bin/bash


.PHONY: run
run: ## run normally (with GPUs)
	-@make clean-results
	@make build
	docker run -it --rm --gpus all \
		-v `pwd`:/work \
		-v `pwd`/$(DIR_DATA):/data \
		$(CONTAINER_NAME) \
		python -O scripts/$(SCRIPT)
# > results/_stdout.txt


.PHONY: debug
debug: ## debug mode (with GPUs)
	@make build
	docker run -it --rm --gpus all \
		-v `pwd`:/work \
		-v `pwd`/$(DIR_DATA):/data \
		$(CONTAINER_NAME) \
		python scripts/$(SCRIPT)


.PHONY: jn
jn: ## run with jupyter 
	@make build
	docker run -it --rm --gpus all \
		-v `pwd`:/work \
		-v `pwd`/$(DIR_DATA):/data \
		-p 8888:8888 \
		$(CONTAINER_NAME) \
		jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root


# -----------
# utils
# -----------
.PHONY: clean-results
clean-results:
	rm -f results/*.png
	rm -f results/*.gif
	

.PHONY:	help
help:	## show help (this)
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'