from pr2_utils import *
class Lidar():
	'''
	  Lidar Class :   A class for importing and pre-processing lidar data.
	  INPUT 
	  lidar_path	  Folder containing lidar point cloud data  
	  OUTPUT 
	  c               sum of the cell values of all the positions hit by range sensor
	'''
	def __init__(self, fpath = 'data/sensor_data/lidar.csv'):
		# Body frame pose
		self.fpath = fpath
		self.bRs = np.array([[0.00130201, 0.796097, 0.605167],\
							[0.999999, -0.000419027, -0.00160026],\
							[-0.00102038, 0.605169, -0.796097]]) # Relative orientation of sensor w.r.t. car
		self.bPs = np.array([0.8349, -0.0126869, 1.76416]).T # Relative pose of sensor w.r.t. car
		# Sensor parameters
		self.range_min = 0.1 # Mimimum radial range of lidar beam 
		self.range_max = 80.0 # Maximum radial range of lidar beam
		self.fov = 190 # Field of View
		self.start_angle = -5 # start_angle of scan relative to lidar heading
		self.end_angle = 185 # end_angle of scan relative to lidar heading
		self.res  = 2/3 # Angular resolution of lidar scan
		# Read lidar data
		data_csv = pd.read_csv(fpath, header=None)
		self.range = data_csv.values[:, 1:]
		self.timestamp = data_csv.values[:, 0]
		# Initialize variables
		self.t = 0

	def get_coords(self):
		'''
		get_coords      Converts lidar point cloud from sensor frame to body frame

		OUTPUT: 
		self.pose       Body frame lidar-point cloud at a time-step self.t
		dtype : float(3,286)
		'''
		# Convert range reading to cartesian coordinates
		angles = np.linspace(self.start_angle, self.end_angle, int(self.fov/self.res) + 1) / 180 * np.pi
		x = self.range[self.t]*np.cos(angles)
		y = self.range[self.t]*np.sin(angles)
		# Body frame transformation
		self.pos = np.stack((x,y,np.zeros((x.shape[0]))), axis = 1) #  (286, 3)
		self.pos = self.bRs@self.pos.T + self.bPs.reshape(3,1) # (3, 3) x (3, 286) -> (3,286)

		# Increase time count
		self.t = self.t + 1
		if self.t >= self.range.shape[0]:
			print("End of sequence!!")
		return self.pos

	def reset(self):
		'''
		  reset			  reset the get_coords routine 
		'''
		self.t = 0
	def advance(self, t):
		'''
		  advance      	  increase the time count variable

		  INPUT:
		  t               increamental quantity
		                  dtype : int

		  OUTPUT:         return TRUE if it's a valid increment
		'''
		self.t += int(t)
		if self.t >= self.range.shape[0]:
			return False
		else:
			return True

def test_lidar():
	ld = Lidar()
	# Visualize the point cloud
	pc = ld.get_coords()
	plt.scatter(pc[0], pc[1])
	plt.show()
	
	# Advance the clock
	ld.advance(12)
	print("Current time step: ", ld.t)

	#Reset
	ld.reset()
	print("Current time step: ", ld.t)


if __name__ == '__main__':
	test_lidar()