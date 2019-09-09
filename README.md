Project 1 - Flocking
====================
**University of Pennsylvania, CIS 565: GPU Programming and Architecture**

Dhruv Karthik: [LinkedIn](https://www.linkedin.com/in/dhruv_karthik/)

Tested on: Windows 10 Home, Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, 16GM, GTX 2070 - Compute Capability 7.5
____________________________________________________________________________________
![Developer](https://img.shields.io/badge/Developer-Dhruv-0f97ff.svg?style=flat) ![CUDA 10.1](https://img.shields.io/badge/CUDA-10.1-yellow.svg) ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/badge/issues-none-green.svg)
____________________________________________________________________________________

| 10,000 Boids | 70,000 Boids |
| ------------- | ----------- |
| ![](images/10KBoids.gif)  | ![](images/70kboids.gif) |

# Runtime Analysis
![FPS w/ Visuals Increasing Num_Boids](images/fpsnumboids_vis.png)
![FPS w/ Visuals Increasing BlockSize](images/fpsblocksize_vis.png)
![FPS w/ NoVisuals Increasing Num_Boids](images/fpsnumboids_novis.png)
![FPS w/ NoVisuals Increasing BlockSize](images/fpsblocksize_novis.png)

# Questions
**For each implementation, how does changing the number of boids affect performance? Why do you think this is?**
On average, more boids decreased the performance. 
1. *Naive*: More boids directly increases the iters of the for loop that iterates over all the boids, thereby increasing runtime
2. *Scattered & Coherent*: While better than the Naive algorithm, both of these algorithms would suffer if several boids were clustered together. High boid density in certain cells would increase the runtime of the for loop iterating over all boids. 

**For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**
Negligible, and uncorellated. However, the coherent grid consistently performed twice as well as the scattered grid. 

**For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**
I expected performance improvements, but not to this level. I was surpised by the fact that merely not indexing into another array could provide (on occasion) almost double the runtime. 
