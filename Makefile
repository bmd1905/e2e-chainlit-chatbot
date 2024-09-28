# Description: Makefile for the project
.PHONY: cl


cl:
	python3 -m chainlit run app.py -w
