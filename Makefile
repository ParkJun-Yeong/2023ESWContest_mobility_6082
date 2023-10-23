# Embedded software Contest 2023
# 전국 짱돌단 연합
# Pose estimation - Kim Hyeong-Sik

menu :
	@echo ''
	@echo '=========================================================================================='
	@echo 'Hello This is YOLO-Pose Estimation toolbox made by KHS'
	@echo 'Before setting you must build virtual venv or conda env with python==3.8 and pytorch==1.x.x'
	@echo '' 
	@echo 'mt	Excute Multi_Thread_serial_connect.py, Real Time with your Webcam(cam_num:0) and Lidar'
	@echo 'mt_data	Excute multi_thread_data.py, It can save real time pose and lidar data (saved .csv)'
	@echo ''
	@echo 'YOU CAN USE THIS AFTER launch roscore($ roscore) AND launch LiDAR($ make ldr)'
	@echo '=========================================================================================='
	@echo ''
	

mt :
	python multi_thread_serial_connect.py

mt_data :
	python multi_thread_data.py

ldr:
	roslaunch rplidar_ros rplidar_a1.launch
