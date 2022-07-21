from pr2_utils import *
class PF_Agent():
    '''
      PF_Agent Class :
                      Particle filter class for system prediction and update
      
      INPUT:
      Map             Map class from map.py

      MEMBERS:
      n               Number of particles
      N_threshold     Threshold on the number of particles
      T               Parallel prediction batch
                      For T = complete motion sequence, particle filter is just a parallelized
                      prediction step implementation. 
                      T_min = 10, since FOG sensor is 10x faster than lidar and encoder scan
      pos             x,y coordinates of the agents
                      dtype : float
                      size : (2 x n x T_total); T_total is complete motion sequence.

      yaw             yaw angle of the agents
                      dtype : float
                      size : (T_total x n); T_total is complete motion sequence.
      
      log             logs agent motion
    '''
    def __init__(self, Map):
        # Map Class
        self.Map = Map
        # Encoder Data
        data_csv = pd.read_csv('data/sensor_data/encoder.csv', header=None)
        self.encoder_data = data_csv.values[:, 1:]
        self.encoder_timestamp = data_csv.values[:, 0]
        # FOG Data
        data_csv = pd.read_csv('data/sensor_data/fog.csv', header=None)
        self.FOG_data = data_csv.values.T[-1].T
        self.FOG_timestamp = data_csv.values[:, 0]
        # Encoder calibrated parameter
        self.encoder_resolution = 4096 
        self.lw_diameter = 0.623479 # meters
        self.rw_diameter = 0.622806 # meters
        self.wheel_base = 1.52439 # meters
        # Other variables
        self.t = 1 # Start time
        self.n = 15 # Number of particles
        self.N_threshold = self.n 
        self.T = 200 # Prediction batch size
        # the particle filter skips T/10 lidar scan updates,
        # but improves implementation speed
        self.alpha = 1/self.n * np.ones((self.n,)) # Particle weights
        self.yaw_angle = np.zeros((self.FOG_timestamp.shape[0], self.n)) # intial yaw angle
        self.pos = np.zeros((2, self.n, self.FOG_timestamp.shape[0]))
        self.log = {}
        self.log["pose"] = self.pos
        self.log["yaw"] = self.yaw_angle
        # Linear and angular velocity
        self.lw_vel = np.diff(self.encoder_data[:, 0], axis = 0)/ \
                        np.diff(self.encoder_timestamp, axis = 0) /self.encoder_resolution \
                        * np.pi * self.lw_diameter * 1e9 # left wheel
        self.rw_vel = np.diff(self.encoder_data[:, 1], axis = 0)/ \
                        np.diff(self.encoder_timestamp, axis = 0) /self.encoder_resolution \
                        * np.pi * self.rw_diameter * 1e9 # right wheel
        self.vel = (self.lw_vel + self.rw_vel) / 2 # Mean velocity
        self.dt = np.diff(self.FOG_timestamp, axis = 0) / 1e9 # Time step
        self.vel = np.interp(self.FOG_timestamp[1:], \
                            self.encoder_timestamp[1:], \
                            self.vel) # Synchronized velocity log

    def dr_prediction(self):
        '''
          dr_prediction   dead reckoning prediction
        '''
        self.yaw_angle = np.cumsum(self.FOG_data[:])
        self.pos = np.array([np.cos(self.yaw_angle[1:]), np.sin(self.yaw_angle[1:])])*(self.dt*self.vel)
        self.pos = np.cumsum(self.pos, axis = 1)
        self.log["pose"] = self.pos
        self.log["yaw"] = self.yaw_angle

    
    def pf_prediction_v0(self):
        '''
          pf_prediction_v0
                          Particle filter prediction, 
                          parallelized implementation over entire motion sequence
        '''
        self.n = 5 # no of particles
        self.sigma_yaw = 0.001 # standard deviation of FOG sensor
        self.sigma_vel = 0.01 # standard deviation of encoder
        self.yaw_angle = np.cumsum(0.0001*np.random.randn(self.FOG_data.shape[0], self.n) +\
                                   np.expand_dims(self.FOG_data[:,2], axis = -1), axis = 0)
        self.pos = np.array([np.cos(self.yaw_angle[1:]), np.sin(self.yaw_angle[1:])]).transpose(0,2,1)*(self.dt)*\
                    (np.expand_dims(self.vel, axis = -1) + self.sigma_vel*np.random.randn(self.FOG_data.shape[0]-1, self.n)).T
        self.pos = np.cumsum(self.pos, axis = -1)

    def pf_prediction(self):
        '''
          pf_prediction
                          Particle filter prediction, 
                          semi-sequential implementation, 
                          prediction over a batch of self.T
        '''
        self.sigma_yaw = 0.001 # standard deviation of FOG sensor
        self.sigma_vel = 0.01 # standard deviation of encoder
        self.yaw_angle[self.t:self.t+self.T] =  self.yaw_angle[self.t-1] + np.cumsum(0.0001*np.random.randn(min(self.T, self.FOG_timestamp.shape[0]-self.T), self.n) + \
                                   np.expand_dims(self.FOG_data[self.t : self.T + self.t], axis = -1), axis = 0)
        pose =  np.expand_dims(self.pos.T[self.t-1].T, axis = -1) + np.cumsum(np.array([np.cos(self.yaw_angle[self.t:self.t+self.T]), np.sin(self.yaw_angle[self.t:self.t+self.T])]).transpose(0,2,1)*(self.dt[self.t:self.T+self.t])* (np.expand_dims(self.vel[self.t:self.T+self.t], axis = -1) + self.sigma_vel*np.random.randn(min(self.T, self.FOG_timestamp.shape[0]-self.T), self.n)).T , axis = -1)
        (self.pos.T[self.t:self.t+self.T]) = pose.T
        
        
    def pf_update(self, lidar_scan):
        '''
          pf_update
                          Sequential implementation of update step
                          Updates the occupancy map and alpha weights
          INPUT:
          lidar_scan      lidar point cloud
                          dtype : float
                          size : 3 x 286
        '''
        self.R = np.block([[np.cos(self.yaw_angle[self.t]), np.sin(self.yaw_angle[self.t])], \
                            [np.sin(self.yaw_angle[self.t]), -np.cos(self.yaw_angle[self.t])]]).reshape(2,2,-1)
        self.p = self.pos.T[self.t] # Position of agent
        lidar_scan = self.R.transpose(2,0,1)@lidar_scan[:2, :] + np.expand_dims(self.p, axis = -1) # World Coordinate lidar_scan
        for i, lscan in enumerate(lidar_scan):
            self.alpha[i] = max(self.Map.map_correlation_v2(lscan).max(),1)*self.alpha[i]
        self.alpha = self.alpha/self.alpha.sum()
        opt_indx = np.argmax(self.alpha) # best particle for map update.
        self.Map.update_map(np.vstack((lidar_scan[opt_indx], np.zeros((1, lidar_scan.shape[-1])))) ,\
                              self.p[opt_indx][0], self.p[opt_indx][1])

    def SLAM(self, lidar):
        '''
          SLAM            SLAM implementation of particle filter
          INPUT:
          lidar           Lidar class from lidar.py
        '''   
        while self.t < self.FOG_timestamp.shape[0]-self.T:
            if self.t%10000 == 1:
                print(int(self.t/self.FOG_timestamp.shape[0]*100), "%% done!") # To check the progress
            self.pf_prediction()
            # if lidar advance is valid then update the map
            if lidar.advance(self.T//10):
                self.pf_update(lidar.get_coords())
            # update alpha weights
            if 1/(self.alpha*self.alpha).sum() < self.N_threshold:
                self.resample()
            # update time step
            self.t = self.t + self.T
        # Visualize the update map
        plt.imshow(self.Map.MAP['map'])
        plt.show(block=True)

    def resample(self):
        '''
          resample        resamples alpha using sample importance resampling method
        '''
        idx = np.repeat(np.arange(self.N_threshold), \
                        np.random.multinomial(self.N_threshold, self.alpha, size=1)[0])
        self.pos.T[self.t] = self.pos.T[self.t][idx]
        self.yaw_angle[self.t] = self.yaw_angle[self.t][idx]
        self.alpha = 1/self.N_threshold*np.ones((self.N_threshold,))
 
def test_pf(ld, mp):
    # Initialize the agent
    agent = PF_Agent(mp)
    # Comment out portions to simulate. Don't run dead reckoning, parallelized prediction
    # and SLAM one after the other!

    # Check pose and map 
    # plt.scatter(agent.log["pose"][0], agent.log["pose"][1])
    # plt.show(block = True)
    # plt.imshow(agent.Map.MAP["map"])
    # plt.show(block = True)

    # Do dead reckoning
    # agent.dr_prediction()
    # plt.scatter(agent.log["pose"][0], agent.log["pose"][1])
    # plt.show(block = True)

    # SLAM!
    agent.SLAM(ld)
    plt.scatter(agent.log["pose"][0], agent.log["pose"][1])
    plt.show(block = True)


if __name__ == '__main__':
    from lidar import Lidar
    from map import Map
    ld = Lidar()
    mp = Map()
    test_pf(ld, mp)