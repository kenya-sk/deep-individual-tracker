# CNN

### src
1. movie2image.cpp
    * get images from movie at arbitrary interval (default: 1 min).

2. movie2trainingData.py
    * load movie, select feature points with a mouse for specific frames,
    and output the coordinate of density map.

3. image2trainingData.py
    * input the directory path, select feature points with a mouse for all image file,
    and output the coordinate of density map.

4. img2localimg.py
	* make local image from original image and cordinate of feature points.

5. cnn.py
    * 6 layer CNN (In Development).
