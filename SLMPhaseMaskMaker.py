"""
SLMPhaseMaskMaker
Lloyd Russell 2015
Implements HOTlab dll (https://github.com/MartinPersson/HOTlab)
"""
from ctypes import CDLL, c_int32, c_float, c_char, c_ushort, c_ubyte, POINTER, byref
from scipy.misc import imread, imsave
import numpy as np
import time
import os

# load targets file, get hot pixel coordinates
target_img_path = "C:/targets/10spotsTight.tif"
target_img = imread(target_img_path)
target_coords = np.nonzero(target_img)  # find the non zero elements (targets)
y = target_coords[0] - 256  # offset, middle of 512,512 image is 0,0
x = target_coords[1] - 256  # offset, middle of 512,512 image is 0,0
N = len(x)  # number of spots
z = np.zeros(N)  # z position of spots, currently all are set to zero
I = target_img[x, y]  # intensity of individual spots

# import the main dll
holo_dll = CDLL("GenerateHologramCUDA.dll")

# create instances of functions
startCUDAandSLM = holo_dll.startCUDAandSLM
GenerateHologram = holo_dll.GenerateHologram
stopCUDAandSLM = holo_dll.stopCUDAandSLM

# set parameters
EnableSLM = 0
TrueFrames = 6
deviceId = 0
h_test = None
x_spots = x
y_spots = y
z_spots = z
I_spots = I
N_spots = N
N_iterations = 10
method = 2
h_pSLMstart = (np.random.random(512*512)-0.5)*2*np.pi  # the starting phase mask (seed)
# h_pSLMstart = np.ones(512*512)

# define c types
# for 'start' function
c_EnableSLM_type = c_int32
c_h_pSLMstart_type = c_float * (512 * 512)
c_LUTfile_type = c_char * 256
c_TrueFrames_type = c_ushort
c_deviceId_type = c_int32
# for 'generate' function
c_h_test_type = c_float * (512 * 512)
c_h_pSLM_type = c_ubyte * (512 * 512)
c_x_spots_type = c_float * N_spots
c_y_spots_type = c_float * N_spots
c_z_spots_type = c_float * N_spots
c_I_spots_type = c_float * N_spots
c_N_spots_type = c_int32
c_N_iterations_type = c_int32
c_h_obtainedAmps_type = c_float * (N_spots * (N_iterations + 1))
c_method_type = c_int32

# make instances of the c types, with values
# for 'start' function
c_EnableSLM = c_EnableSLM_type(EnableSLM)
c_h_pSLMstart = c_h_pSLMstart_type()
c_LUTfile = c_LUTfile_type()
c_TrueFrames = c_TrueFrames_type(TrueFrames)
c_deviceId = c_deviceId_type(deviceId)
# for 'generate' function
c_h_test = h_test
c_h_pSLM = c_h_pSLM_type()
c_x_spots = c_x_spots_type()
c_y_spots = c_y_spots_type()
c_z_spots = c_z_spots_type()
c_I_spots = c_I_spots_type()
c_N_spots = c_N_spots_type(N_spots)
c_N_iterations = c_N_iterations_type(N_iterations)
c_h_obtainedAmps = c_h_obtainedAmps_type()
c_method = c_method_type(method)

# fill c arrays with values
for i in range(N_spots):
    c_x_spots[i] = x_spots[i]
    c_x_spots[i] = x_spots[i]
    c_y_spots[i] = y_spots[i]
    c_z_spots[i] = z_spots[i]
    c_I_spots[i] = I_spots[i]

for i in range(512*512):
    c_h_pSLMstart[i] = h_pSLMstart[i]

# set argument types for functions
startCUDAandSLM.argtypes = [c_EnableSLM_type,
                            POINTER(c_h_pSLMstart_type),
                            POINTER(c_LUTfile_type),
                            c_TrueFrames_type,
                            c_deviceId_type]
GenerateHologram.argtypes = [POINTER(c_h_test_type),
                             POINTER(c_h_pSLM_type),
                             POINTER(c_x_spots_type),
                             POINTER(c_y_spots_type),
                             POINTER(c_z_spots_type),
                             POINTER(c_I_spots_type),
                             c_N_spots_type,
                             c_N_iterations_type,
                             POINTER(c_h_obtainedAmps_type),
                             c_method_type]

# run the functions (start, make phase mask, stop)
error1 = startCUDAandSLM(c_EnableSLM, byref(c_h_pSLMstart), byref(c_LUTfile),
                         c_TrueFrames, c_deviceId)
if error1 != 0:
    print(error1)

start_time = time.time()
error2 = GenerateHologram(c_h_test, byref(c_h_pSLM), byref(c_x_spots),
                          byref(c_y_spots), byref(c_z_spots), byref(c_I_spots),
                          c_N_spots, c_N_iterations, byref(c_h_obtainedAmps),
                          c_method)
elapsed = time.time() - start_time
print('Phase mask made in: ' + str(elapsed) + ' s')
if error2 != 0:
    print(error2)

error3 = stopCUDAandSLM()
if error3 != 0:
    print(error3)

# convert results from c_array to numpy
h_pSLM_8b = np.ctypeslib.as_array(c_h_pSLM)  # the hologram

h_pSLM_16b = h_pSLM_8b.astype(np.uint16)
h_pSLM_16b = h_pSLM_16b * 256
h_pSLM_16b = np.reshape(h_pSLM_16b, [512, 512])

# save
# make directories
directory, filename = os.path.split(target_img_path)
filename_noext = os.path.splitext(filename)[0]
today_date = time.strftime('%Y%m%d')
savedir = os.path.join(directory, filename_noext + "_" + today_date +
                       "_CUDAphase")
if not os.path.exists(savedir):
    os.makedirs(savedir)

# save phase mask
savename = os.path.join(savedir, filename_noext + "_" + today_date +
                        "_CUDAphase_" + "m" + str(method) + "_i" +
                        str(N_iterations) + ".tiff")
imsave(savename, h_pSLM_16b)

# save reconstructed targets image for sanity check
reconstructed_targets_img = np.zeros([512, 512], dtype=np.uint16)
reconstructed_targets_img[y+256, x+256] = 255
savename = os.path.join(savedir, filename_noext + "_" + today_date +
                        "_TargetImg" + ".tiff")
imsave(savename, reconstructed_targets_img)
