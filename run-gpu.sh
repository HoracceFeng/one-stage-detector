sudo nvidia-docker run -it $1 \
	-v /HDATA/6/01366808/hnfeng/code/detector:/code \
	-v /HDATA/6/01366808/hnfeng/data:/data \
        --ipc="host" \
	cuda9.0-cudnn7.0-tf1.5-keras2.1-pytorch1.0-opencv3.4-py3.6:latest bash
