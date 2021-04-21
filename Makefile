
build: 
	docker build -t endrelaszlo/gaussian_process -f Dockerfile_custom .

run:
	docker run -p 9999:9999 -it --rm endrelaszlo/gaussian_process

