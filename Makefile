
build: 
	docker build -t endrelaszlo/gaussian_process -f Dockerfile_custom .

run:
	docker run -p 8888:8888 -p 8050:8050 -it --rm endrelaszlo/gaussian_process

