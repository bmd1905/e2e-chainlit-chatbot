# Description: Makefile for the project
.PHONY: cl litellm

cl:
	python3 -m chainlit run main.py -w

litellm:
	sudo docker-compose up --build
