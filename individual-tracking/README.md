# Deep Individual Tracker / Indeividual Tracking
## Introduction
Tracking is performed based on the location of the individual detected by 'indevidual-detection'. When tracking, template matching is performed for the neighborhood of the detection point and correspondence is made between frames.

## Execution Environment
The execution environment can be build using Docker Comopse. In this case, OpenCV version 3.4.0 is used for execution. The directory of `/data` was mounted by docker-compose.

``` bash
# launch countainer
$ docker-compose exec tracker /bin/bash
```

## Execute Tracking
To perform tracking, two files are required in advance. One file containing the video and the coordinates of the individual detected frame by frame. For detection of individuals can be done using [the detection algorithm included in this repository](https://github.com/kenya-sk/deep-individual-tracker/tree/master/individual-detection), sbut other algorithms can be used as long as the files are in the same format. If the two file are ready, set the paths to each file in the config file (conf/trackinf_config.cfg).

Once the configuration is complete and the compiled main file is run, tracking is executed. The following two results are saved as a result of the tracking. The first result is a video file in which the tracked trajectory is drawn on the input video (the destination is the value set in the "output_video_path" parameter of the config). The second result is the statistics computed from the tracking results (the destination is the value set in the "output_stats_dirctory" parameter of the config)). It records the mean, variance, and maximum of the amount of movement in one second.

```
# comple main file
cd src
make all

# excute tracking
./main
```
