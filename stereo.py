from pr2_utils import *
class Stereo():
	def __init__(self):
		self.cam_mat = np.array([[8.1690378992770002e+02, 5.0510166700000003e-01, 6.0850726281690004e+02,],\
								[0., 8.1156803828490001e+02, 2.6347599764440002e+02],\
								[ 0., 0., 1.]]) # Camera Instrinsic Matrix
		self.R = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02], \
							[0., 7.7537235550066748e+02 , 2.5718049049377441e+02], \
							[0., 0., 1.]]) # Rotation Matrix
		self.Ro = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) # Flip Matrix      
		self.ds = 0.475143600050775*8.1378205539589999e+02 # Depth scale : f x su x b

	def read_stereo(self, path_l = 'data/image_left.png', path_r = 'data/image_right.png'):

		self.image_l = cv2.imread(path_l, 0)
		self.image_r = cv2.imread(path_r, 0)

		self.image_l = cv2.cvtColor(self.image_l, cv2.COLOR_BAYER_BG2BGR)
		self.image_r = cv2.cvtColor(self.image_r, cv2.COLOR_BAYER_BG2BGR)

		image_l_gray = cv2.cvtColor(self.image_l, cv2.COLOR_BGR2GRAY)
		image_r_gray = cv2.cvtColor(self.image_r, cv2.COLOR_BGR2GRAY)

		stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
		self.disparity = stereo.compute(image_l_gray, image_r_gray)

		fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
		ax1.imshow(self.image_l)
		ax1.set_title('Left Image')
		ax2.imshow(self.image_r)
		ax2.set_title('Right Image')
		ax3.imshow(self.disparity, cmap='gray')
		ax3.set_title('Disparity Map')
		plt.show(block=True)

	def biv_texture(self):
		nx, ny = self.disparity.shape
		depth = self.ds/np.clip(self.disparity, 1, 255)
		XX, YY = np.meshgrid(np.linspace(-nx/2, nx/2, nx), np.linspace(-ny/2, ny/2, ny))
		self.UVs = np.vstack((YY.reshape(-1), XX.reshape(-1), np.ones((XX.reshape(-1).shape[0])))).T
		pp = (self.R)@((np.linalg.inv(self.cam_mat)@self.UVs.T)*depth.reshape(-1))
		plt.scatter(pp[0], pp[2], c= self.image_l.T.reshape(-1,3)/255)
		plt.show(block=True)

if __name__ == '__main__':
	cam = Stereo()
	cam.read_stereo()
	cam.biv_texture()