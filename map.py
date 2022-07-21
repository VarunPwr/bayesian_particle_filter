from pr2_utils import *
class Map():
    '''
      Map Class :     A class for creating, storing and processing occupancy map
      
      MEMBERS:
      MAP             Dictionary that stores the map and it's attributes
      x_range         Perturbation in X axis
      y_range         Perturbation in Y axis
      logit           prob(m = 1) / (1 - prob(m = 1))
      x_im            X pixel pose of the map
      y_im            Y pixel pose of the map
    '''
    def __init__(self):
        # initialize MAP dictionary
        self.MAP = {}
        self.MAP['res']   =  1 #meters
        self.MAP['xmin']  = -200  #meters
        self.MAP['ymin']  = -1400
        self.MAP['xmax']  =  1400
        self.MAP['ymax']  =  200 
        self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        self.y_im = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map
        self.x_range = np.arange(-4,4+1,1) # Add perturbation
        self.y_range = np.arange(-4,4+1,1) # Add perturbation
        self.logit = 4 # Not used here
        # Every coordinate in the map is occupied.
        self.MAP['map'] = np.ones((self.MAP['sizex'],self.MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    
    def update_map(self, coords, x_o = 0, y_o = 0, pose = np.eye(4)):
        '''
          udpate_map :    Updates the occupancy map with lidar point cloud
          
          INPUT:

          coords          Lidar points 
                          size : 3 x 286
                          dtype : float
          pose(optional)  Body to world frame transformation matrix
                          size : 4 x 4
                          dtype : float
          
          x_o, y_o(optional)
                          World frame body pose
        '''
        w_coords = pose@np.vstack((coords, np.ones(coords.shape[1]))) # World frame.
        # Zero-indexed grid points
        xis = np.ceil((w_coords[0, :] - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((w_coords[1, :] - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        # Grid coordinates of lidar origin
        xois = np.ceil((x_o - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        yois = np.ceil((y_o - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        indx = np.array([[xois],[yois]])
        
        # Calculate bresenham2d distance.
        for t in range(xis.shape[0]):
            indx = np.hstack((indx, bresenham2D(xois, yois, xis[t], yis[t])))
        # Only keep unique indices
        indx = np.unique(indx.T, axis = 0) # (No of updates, 2)
        # Update the map
        indx = indx.astype(int)
        for ind in indx:
            self.MAP['map'][ind[0]][ind[1]] = 0

    def map_correlation_v2(self, lidar_scan):
        '''
          map_correlation_v2 
                          Finds the correlation between the point cloud and the map
                          Parallelized implementation, also available in utils file
          
          INPUT:

          lidar_scan      Global frame lidar point cloud coordinates 
                          size : 3 x 286
                          dtype : float
          
          OUTPUT:
          correlation_mat Correlation matrix
                          size : 9 x 9
                          dtype : float
        '''
        nx, ny = self.MAP['map'].shape
        
        nxs = self.x_range.size
        nys = self.y_range.size
        cpr = np.zeros((nxs, nys))
        y_range = np.expand_dims(self.y_range, axis=1)
        x_range = np.expand_dims(self.x_range, axis=1)
        y1 = lidar_scan[1,:] + y_range
        x1 = lidar_scan[0,:] + x_range
        iy = np.int16(np.round((y1-self.MAP['ymin'])/self.MAP['res']))
        ix = np.int16(np.round((x1-self.MAP['xmin'])/self.MAP['res']))
        ix = np.tile(ix.reshape(-1),nys)
        iy = np.tile(iy,nxs).reshape(-1)
        valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                              np.logical_and((ix >=0), (ix < nx)))
        return np.ma.masked_array(self.MAP['map'][ix%nx, iy%ny],1- valid).reshape(-1,lidar_scan[1,:].shape[0]).sum(axis = -1).reshape(nxs, nys).T

    def refresh_map(self):
        '''
          refresh_map     Reintializes the map
        '''
        self.MAP['map'] = np.ones((self.MAP['sizex'],self.MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8


def test_map(ld):
    map = Map()
    # Visualize map
    plt.imshow(map.MAP['map'])
    plt.show(block=True)
    # Update map
    map.update_map(coords=ld.get_coords())
    # Check output
    plt.imshow(map.MAP['map'])
    plt.show(block=True)
    # Refresh map
    map.refresh_map()
    plt.imshow(map.MAP['map'])
    plt.show(block=True)


if __name__ == '__main__':
    from lidar import Lidar
    ld = Lidar()
    test_map(ld)
