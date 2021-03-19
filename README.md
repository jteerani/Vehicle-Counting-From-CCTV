# Vehicle-Counting-From-CCTV
![alt text](https://github.com/jteerani/Vehicle-Counting-From-CCTV/blob/main/output_1.png)

1. yolo_custom.ipynb --> which is used for training custom data in YOLOV4.
		                ---> I trained via Google CoLab  
			
  ******If you want a custom data,please contact me********
  
2. yolo-video-vehicle-count.py --> which is used for counting vehicle.
3. sort.py --> which is tacker algorithm.This program is free software: https://github.com/abewley/sort
5. model --> This folder is weight and configuration for custom data that I used for training yolov44
	- weight can download from https://drive.google.com/file/d/1-0I6Dnd90F2Wt24Vt_XjlSS-FUwVfPPY/view?usp=sharing
7. input --> Example input video file.
	- can download from https://drive.google.com/file/d/1sBI0RqDmLltTgpr9RSadMlVouXdXob9O/view?usp=sharing
9. chart_custom_yolov4.png --> This picture is final output chart from custom training yolov4  

###### Result of training custom data in YOLOV4
![alt text](https://github.com/jteerani/Vehicle-Counting-From-CCTV/blob/main/chart_custom_yolov4.png)


###### This project is designed for CCTV in specific area so that it need to be improved with more complex custom data and complex perspective.
