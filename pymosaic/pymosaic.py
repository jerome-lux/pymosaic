import numpy as np
from PIL import Image
import os
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import json
from random import shuffle
import shutil
import numba as nb
from scipy.spatial import cKDTree

#TODO: add an opacity setting
#Add a script to make a mosaic in one step.
#Note: cKDTree not faster than numba (up to 40k images).

VALID_IMAGE_FORMATS = ['jpg','jpeg','png','bmp','tif']

@nb.njit(error_model='numpy',parallel=True)
def average_pooling(img,steps,stepsize):
    """steps give the number of steps in x and y
    stepsize are the tile size in both directions
    currently works for RGB image only
    """
    out = np.empty((steps[0],steps[1],3))
    
    for c in nb.prange(img.shape[2]):
        for i in nb.prange(0, steps[0]):
            for j in nb.prange(0, steps[1]):
                x0 = i*stepsize[0]
                x1 = min((i+1)*stepsize[0],img.shape[0])
                y0 = j*stepsize[1]
                y1 = min((j+1)*stepsize[1],img.shape[1])
                out[i,j,c]=np.mean(img[x0:x1, y0:y1,c])

    return out  

def _resize(imfile,datadict,target_dir,size=(64,64),order=2,optimisation=3):

    image = Image.open(imfile)
    image = image.resize(size=size, resample=order, reducing_gap=optimisation)
    image.save(os.path.join(target_dir,os.path.basename(imfile)))
    npim = np.array(image)
    datadict[os.path.basename(imfile)] = [npim[:,:,0].mean(),npim[:,:,1].mean(),npim[:,:,2].mean()]
    return datadict

def compute_RGB_data(imfile,datadict,input_dir):

    npim = np.array(Image.open(imfile))
    datadict[os.path.relpath(imfile,input_dir)] = [npim[:,:,0].mean(),npim[:,:,1].mean(),npim[:,:,2].mean()]
    return datadict

def create_tileset(input_dir, target_dir, size=(64, 64), max_im=None, replace=False, ncpu=cpu_count()):

    """Resize images in directory r (recursive search, so all images in subfolder are processed)
    Put them in target_dir 
    Save average RGB values for each images in RGBdata.json. If file already exist, it just add new data to it
    return dict of resized image
    """

    tileset = [os.path.join(r, f) for r, d, filenames in os.walk(input_dir) for f in filenames 
               if os.path.splitext(f)[-1][1:].lower() in VALID_IMAGE_FORMATS]

    shuffle(tileset)
    if max_im is not None:
        tileset = tileset[:max_im]
    
    if replace:
        try:
            shutil.rmtree(target_dir)
        except:
            pass
    
    os.makedirs(target_dir,exist_ok=True)
    
    manager = Manager()
    imdata = manager.dict()
    f = partial(_resize,datadict=imdata,target_dir=target_dir,size=size)
    with Pool(ncpu) as pool:
        pool.map(f,tileset)
        pool.close()
        pool.join()
    
    print( "{} images have been resized and saved to {}".format(len(tileset),target_dir) )
    print( "Saving RGB data to {}".format(os.path.join(target_dir,'RGBdata.json')))
    
    with open(os.path.join(target_dir,'RGBdata.json'), 'w') as outfile:
        json.dump(imdata.copy(), outfile) 
    
    return imdata

def create_RGB_stats(input_dir,ncpu=cpu_count()):
    
    """Compute RGB average per channel in al images in input_dir
    Save stats in RGBdata.json in input_dir
    If RGBdata.json already exists, it just adds new data to it
    """

    images = [os.path.join(r, f) for r, d, filenames in os.walk(input_dir) for f in filenames 
               if os.path.splitext(f)[-1][1:].lower() in VALID_IMAGE_FORMATS]
    
    manager = Manager()
    imdata = manager.dict()
    f = partial(compute_RGB_data,datadict=imdata,input_dir=input_dir)
    with Pool(ncpu) as pool:
        pool.map(f,images)
        pool.close()
        pool.join()
    
    RGBdatafile = os.path.join(input_dir,'RGBdata.json')
    
    # if os.path.exists(RGBdatafile):
    #     os.remove(RGBdatafile)
    
    with open(RGBdatafile, 'w') as outfile:
        json.dump(imdata.copy(), outfile) 
    
    return imdata

@nb.njit(parallel=True)
def _get_best_match(x,keylist,RGBarray):
    
    n = len(keylist)
    dist = np.zeros(n).astype(np.float64)
            
    for k in nb.prange(n):
        dist[k] = np.sum((x-RGBarray[k,:])**2)
    
    # print(np.argmin(dist))
    return keylist[np.argmin(dist)]   


class MosaicMaker():
    
    def __init__(self,input_image,tiles_dir,target_image=None,tilesize=(50,50),mintiles=100):
        
        """tiles_dir: directory where the tiles are stored
        input_image: path to the input image to be "mosaicified"
        target_image: name of the mosaic iamage (default= "mosaic-" + input_image name)
        tilesize: size of tiles. Resized if needed
        mintiles: minimum number of tiles in a given direction. The size of the mosaic image is computed accordingly
        """
        
        self._tiles_dir = tiles_dir
        self._input_image_filename = input_image
        self._input_image = np.array(Image.open(input_image).convert('RGB'))
        print("Input image size:",self._input_image.shape)
        self._tilesize = tilesize
        self._mintiles = mintiles
        
        #Get tiles (R,G,B) average values for each image in the tile directory
        self.get_tiles_stats(self._tiles_dir)
        
        self.image_stats_computed = False
          
    @property
    def input_image_filename(self):
        return self._input_image_filename

    @property
    def tilesize(self):
        return self._tilesize
    
    @property
    def mintiles(self):
        return self._mintiles
    
    @property
    def tiles_dir(self):
        return self._tiles_dir
    
    @input_image_filename.setter
    def input_image_filename(self,input_image):
        self._input_image_filename = input_image
        self._input_image = np.array(Image.open(input_image).convert('RGB'))
        self.image_stats_computed = False
    
    @tilesize.setter
    def tilesize(self,tilesize):
        self._tilesize = tilesize
        self.image_stats_computed = False
    
    @tilesize.setter
    def mintiles(self,mintiles):
        self._mintiles = mintiles
        self.image_stats_computed = False
    
    @tiles_dir.setter
    def tiles_dir(self,tiles_dir):
        self._tiles_dir = tiles_dir
        self.get_tiles_stats(self._tiles_dir)
    
    def compute_image_stats(self,save=False):
                
        r = self._tilesize[0] /self._tilesize[1]
        
        #Compute the size of tiles to get at leats mintiles number of tiles in both directions
        sx = self._input_image.shape[0] // self._mintiles
        sy = self._input_image.shape[1] // self._mintiles
        
        if r > 1:
            sx = int(np.round(r * sy))
        else:
            sy = int(np.round(r * sx))
                
        #Get the number of tiles
        self.tiles = (self._input_image.shape[0] // sx,
                      self._input_image.shape[1] // sy)
        
        print("Computing average RGB values of input image in {}x{} boxes".format(self.tiles[0],self.tiles[1]))
        
        #Compute the tile size in the input_image
        self._input_image_tilesize = (self._input_image.shape[0] // self.tiles[0],
                                     self._input_image.shape[1] // self.tiles[1])
        
        # Average pooling using tilesize -> each pixel store the R,G,B average in each tile.     
        self.pooled_image = average_pooling(self._input_image,self.tiles,
                                            self._input_image_tilesize)
    
        self.image_stats_computed = True    
    
    def get_tiles_stats(self,tiles_dir): 
        """Get the RGB data of the tileset
        If RGBdata.json does not exist in the tiles directory, it is computed
        """
        
        print("Get RGB values of tiles")
        
        if not os.path.exists(os.path.join(tiles_dir,'RGBdata.json')):
            print("Computing RGB stats of tile images")
            create_RGB_stats(tiles_dir)
         
        with open(os.path.join(tiles_dir,'RGBdata.json'),'r') as f:
            self.tilesdata = json.load(f)  
            
        self.kdtree = cKDTree(list(self.tilesdata.values()))
       
    def build_mosaic(self,filename=None,reuse=0,randomize=True):
        
        """parallel implementation is slower if tile pool size is small
        """
        #Compute average (R,G,B) values in boxes in original image if needed
        if not self.image_stats_computed:
            self.compute_image_stats()
        
        mosaic_shape = (self.tiles[0] * self._tilesize[0],
                        self.tiles[1] * self._tilesize[1],
                        3)
        
        print("Mosaic will be made of {}x{} = {} tiles".format(self.tiles[0],self.tiles[1],np.prod(self.tiles)))
        print("Its size in pixels will be {}x{}".format(mosaic_shape[0],mosaic_shape[1]))
                        
        self.mosaic_image = np.zeros(mosaic_shape).astype(np.uint8)
        
        if np.prod(self.tiles) > len(self.tilesdata):
            print("Not enough tiles to make mosaic without reuse")

       # Using numba
        totsize = np.prod(self.tiles)
        RGBarray = np.array(list(self.tilesdata.values()))
        # keylist = nb.typed.List(self.tilesdata.keys())
        keylist = list(self.tilesdata.keys())

        if reuse < len(self.tilesdata.values()):
            tilescounter = np.zeros(len(self.tilesdata.values()))

        #Assemble tiles
        k = 0
        h, w = self.pooled_image.shape[:-1]
        #Iterate randomly so that best tile matches are not always at the same place
        if randomize:
            randomRange = np.arange(h*w)
            np.random.shuffle(randomRange)
        else:
            randomRange = range(h*w)
            
        for n in randomRange:
            i,j = divmod(n, w)
            k += 1
            print("Assembling tiles: {:04.1f}%".format(100 * k / totsize), flush=True, end='\r')
            # key = _get_best_match(self.pooled_image[i,j,:],keylist,RGBarray)
            index = self.kdtree.query(self.pooled_image[i,j,:])
            key = keylist[index[1]]
            #Deleting the item in tilesdata is too slow. Just add a big value to all RGB values so that image won't be chosen
            if reuse < len(self.tilesdata):
                ind = keylist.index(key)
                tilescounter[ind] += 1
                if tilescounter[ind] > reuse:
                    RGBarray[ind,:] += 1024 
            try:
                self.mosaic_image[i*self._tilesize[0]:(i+1)*self._tilesize[0],
                                j*self._tilesize[1]:(j+1)*self._tilesize[1],
                                :] = np.array(Image.open(os.path.join(self._tiles_dir,key)))
            except:
                print("Problem loading and inserting tile {}. \n Check that the images in tiles directory are all of size {}".format(key,self._tilesize))
                
        #Saving mosaic
        if filename is None:
            filename = "mosaic-reuse{}-{}x{}-TS{}x{}-{}".format(
                reuse,
                self.tiles[0],
                self.tiles[1],
                self._tilesize[0],
                self._tilesize[1],
                os.path.basename(self._input_image_filename))

            if randomize:
                filename = 'Randomized-' + filename
                
        PILim = Image.fromarray(self.mosaic_image)
        PILim.save(filename)
        
        print("Mosaic saved in ",filename)
        
        return self.mosaic_image      
        
    
