Author : Varun Pawar
Email : vpawar@ucsd.edu

File strucutre			Description 
-- pr2_utils.py			Contains utility files for the project
-- lidar.py				Lidar class which is used for importing and anayzling lidar point cloud
-- map.py				map class for generating and updating occupancy map
-- pf_agent.py			Particle filter agent class for particle filter SLAM routine
-- stereo.py			Stereo class for disparity map generation, bird's eye view texture generation

How to run lidar.py?
$python lidar.py
-- Description: Contains lidar class. The above command will import a lidar sensor data from the csv data file and
transform the data into cartesian coordinates. It is then visualized using a scatter plot.

How to visualize and update map?
$python map.py
-- Description: Creates a map class. The map class will generate a map array object which can be visualized. It will also use
lidar class for updating the occupancy map.

How to run dead-reckoning sequence?
$python pf_agent.py
-- Description: Creates a pf_agent class. It contains dead reckoning routine(which is commented out!). It will generate the trajectory and display a graph.

How to run particle filter SLAM?
$python pf_agent.py
-- Description: Create a pf_agent class. It contains SLAM routine(which is not commented). You can change number of particles and prediction batch size in the code (self.T and self.n) for different variations of the SLAM routine. 

How to genereate disparity map and biv image?
$python stereo.py
-- Description: It creates a stereo class. Stereo class is used to create a disparity map nad biv image. You can change the file path for different example. 