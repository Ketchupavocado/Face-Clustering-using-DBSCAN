# Live Clustering using DBSCAN
The idea of my project is an easy to use facial detection and recognition script that can be understood and used within minutes. With it's simple and smooth live feed and 2 different variations of the script for both CPU and GPU, it can be useful to everybody.


## Final Project Video

https://github.com/user-attachments/assets/4c295864-2bd9-42ad-9480-d1df834fabb2

My code runs off of a facial recognition and detection model called Face Clustering using DBSCAN. The changes I made was, GPU and USB camera compatibility, live streaming, detecting within a min/max range, easy fps changer, bounding boxes that smoothly track with faces, and automatic screenshots and learning of unknown faces within 2 runs of the script, not only is it very simple but is easy to use and change to fit your name and face.

## Requirements

**Base Requirements(CPU)**
```bash
pip install dlib@https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
pip install face-recognition==1.3.0
pip install opencv-python==4.5.2.52
pip install scikit-learn==0.24.2
```
**Additional Requirements(GPU)**
```
pip install Flask==2.0.1
pip install numpy==1.20.3

```
### How to use my code

**Using the CPU version**
To use the CPU version of the script, open up a python engine of your choice and paste the script. In a .venv terminal, make sure you have all dependencies installed which should all be above in the requirements section of the README. Once this is completed, simply run the script and you should see a new app of the live feed from your camera appear. When a face is shown in the camera and is within a certain distance of the camera, it will begin to track and follow your face, which should be detected as unknown. Soon, you should see in the console that images have been taken, and moved to a directory named Unknown_#. After clicking Q the code should instantly stop running and after renaming the unknown directory, your face should be saved and ready to be recognized from now on. Within just a few steps you have a fully functioning recognition model.

**Using the GPU version**
To use the GPU version, it's a little bit more complex and some additional dependencies will be required, once the requirements are fulfilled in the SSH terminal of your NVIDIA GPU. Simply type python3 j and press tab. The full command should have appeared, after pressing enter you should be linked to a new tab where your live feed can be easily accessed, even from your phone or another mobile device. Since this is completely headless you can do this almost anywhere. Once this is done, the USB Camera linked to your Jetson should boot up and begin to track for faces. Once unknown faces are detected, it will begin to rapidly take photos and generate a large variety of folders. Using an app such as WinSCP to preview the images, you can choose a folder that has images you liked and you can type in your terminal: mv ClusteredFaces/Unknown_#(The # of the folder you liked) ClusteredFaces/(Your name or what you want to call yourself). Once you've done this, just deleted the remaining unknown folders and bam, its ready to run and fully operational!

### Thank you so much for using/checking out my project!
