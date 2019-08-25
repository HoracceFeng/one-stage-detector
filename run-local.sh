sudo docker run -it $1 \
	-v /home/horacce/ERA/home/horacce/NOAH/LEARN/Detector_Pytorch/detector:/code \
	-v /home/horacce/ERA/home/horacce/NOAH/DATA/traffic-cognition/TRAIN-MINI:/data \
        --ipc="host" \
	pytorch1.1-cpu_py3.7_cv4.1:latest bash
