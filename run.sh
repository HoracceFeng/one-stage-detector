sudo docker run -it $1 \
	-v /Users/horacce/Nirva/Project/Detect-Pytorch/detector:/code \
	-v /Users/horacce/Nirva/Data/:/data \
	pytorch1.1-cpu_py3.7_cv4.1:latest bash
