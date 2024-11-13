import numpy as np
from PIL import Image
import os
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import json
from random import shuffle
import shutil
import numba as nb
from copy import copy, deepcopy
from scipy.spatial import cKDTree

# TODO: regenerate tiles stats after building mosaic
# TODO: add an opacity setting
# TODO: add the possibility to update the kdtree periodically to limit
# tile reuse (faster than brute-force for not too large set ?)

EPS = 0.05
DEF_BUCKET_SIZE = 10000
VALID_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tif']
LCONV = np.array([0.2126, 0.7152, 0.0722])


@nb.njit(error_model='numpy', parallel=True)
def average_pooling(img, steps, stepsize):
    """steps give the number of steps in x and y
    stepsize are the tile size in both directions
    currently works for RGB image only
    """
    out = np.empty((steps[0], steps[1], 3))

    for c in nb.prange(img.shape[2]):
        for i in nb.prange(0, steps[0]):
            for j in nb.prange(0, steps[1]):
                x0 = i * stepsize[0]
                x1 = min((i + 1) * stepsize[0], img.shape[0])
                y0 = j * stepsize[1]
                y1 = min((j + 1) * stepsize[1], img.shape[1])
                out[i, j, c] = np.mean(img[x0:x1, y0:y1, c])

    return out


def _resize(imfile, datadict, target_dir, size=(64, 64), order=2, crop_aspect_ratio=True, optimisation=3.0):

    image = Image.open(imfile)
    ar = size[1] / size[0] # width / height
    h, w = image.height, image.width

    # Crop image
    if crop_aspect_ratio is True:
        target_w = min(int(np.around(h * ar)), w)
        target_h = int(np.around(target_w / ar))
        ch = h - target_h
        cw = w - target_w
        image = image.crop((cw // 2 + cw % 2,
                            ch // 2 + ch % 2,
                            w - cw // 2,
                            h - ch // 2))
    if image.mode != 'RGB' and image.mode != 'L':
        image = image.convert('RGB')

    image = image.resize(size=(size[1], size[0]), resample=order, reducing_gap=optimisation)
    filename = os.path.splitext(os.path.basename(imfile))[0] + ".jpg"
    image.save(os.path.join(target_dir, os.path.basename(filename)), format="JPEG", mode='RGB')

    npim = np.array(image)
    if len(npim.shape) == 2 :
        m = npim[:, :].mean()
        datadict[os.path.basename(filename)] = [m, m, m]
    else:
        datadict[os.path.basename(filename)] = [npim[:, :, 0].mean(), npim[:, :, 1].mean(), npim[:, :, 2].mean()]
    return datadict



def compute_RGB_data(imfile, datadict, input_dir):

    npim = np.array(Image.open(imfile))

    if len(npim.shape) == 2 :
        m = npim[:, :].mean()
        datadict[os.path.relpath(imfile, input_dir)] = [m, m, m]
    else:
        datadict[os.path.relpath(imfile, input_dir)] = [npim[:, :, 0].mean(), npim[:, :, 1].mean(), npim[:, :, 2].mean()]
    return datadict


def create_tileset(input_dir, target_dir, size=(64, 64), crop_aspect_ratio=True, max_im=None, replace=False, ncpu=cpu_count()):
    """Resize images in directory r (recursive search, so all images in subfolder are processed)
    Put them in target_dir
    Save average RGB values for each images in RGBdata.json. If file already exist, it just add new data to it
    return dict of resized image
    size is (height, width) i.e. numpy (nx, ny)
    """

    tileset = [os.path.join(r, f) for r, d, filenames in os.walk(input_dir) for f in filenames
               if os.path.splitext(f)[-1][1:].lower() in VALID_IMAGE_FORMATS]

    shuffle(tileset)
    if max_im is not None:
        tileset = tileset[:max_im]

    if replace:
        try:
            shutil.rmtree(target_dir)
        except BaseException:
            pass

    os.makedirs(target_dir, exist_ok=True)

    manager = Manager()
    imdata = manager.dict()
    f = partial(_resize, datadict=imdata, target_dir=target_dir, size=size, crop_aspect_ratio=crop_aspect_ratio)
    with Pool(ncpu) as pool:
        pool.map(f, tileset)
        pool.close()
        pool.join()

    print("{} images have been resized and saved to {}".format(len(tileset), target_dir))
    print("Saving RGB data to {}".format(os.path.join(target_dir, 'RGBdata.json')))

    with open(os.path.join(target_dir, 'RGBdata.json'), 'w') as outfile:
        json.dump(imdata.copy(), outfile)

    return imdata


def create_RGB_stats(input_dir, ncpu=cpu_count()):
    """Compute RGB average per channel in al images in input_dir
    Save stats in RGBdata.json in input_dir
    If RGBdata.json already exists, it just adds new data to it
    """

    images = [os.path.join(r, f) for r, d, filenames in os.walk(input_dir) for f in filenames
              if os.path.splitext(f)[-1][1:].lower() in VALID_IMAGE_FORMATS]

    print("Found {} images in {}".format(len(images), input_dir))

    manager = Manager()
    imdata = manager.dict()
    f = partial(compute_RGB_data, datadict=imdata, input_dir=input_dir)
    with Pool(ncpu) as pool:
        pool.map(f, images)
        pool.close()
        pool.join()

    RGBdatafile = os.path.join(input_dir, 'RGBdata.json')

    # if os.path.exists(RGBdatafile):
    #     os.remove(RGBdatafile)

    with open(RGBdatafile, 'w') as outfile:
        print("Saving tiles RGB values in  {}".format(len(RGBdatafile)))
        json.dump(imdata.copy(), outfile)

    return imdata

# 0.2126 * R + 0.7152 * G + 0.0722 * B
@nb.njit(parallel=True)
def _get_best_match_with_luminance(x, keylist, RGBarray):

    n = len(keylist)
    dist = np.zeros(n).astype(np.float64)

    for k in nb.prange(n):
        target = np.zeros((4))
        target[0:3] = x
        target[-1] = np.sum(LCONV * x)
        data = np.zeros((4))
        data[0:3] = RGBarray[k, :]
        data[-1] = np.sum(LCONV * RGBarray[k, :])
        dist[k] = np.sum((target - data)**2)

    # print(np.argmin(dist))
    return keylist[np.argmin(dist)]

@nb.njit(parallel=True)
def _get_best_match(x, keylist, RGBarray):

    n = len(keylist)
    dist = np.zeros(n).astype(np.float64)

    for k in nb.prange(n):
        dist[k] = np.sum((x - RGBarray[k, :])**2)

    # print(np.argmin(dist))
    return keylist[np.argmin(dist)]


class MosaicMaker():

    def __init__(self, input_image, tiles_dir, target_image=None,
                 tilesize=(50, 50), mintiles=100, method='brute-force',
                 use_luminance=False):
        """tiles_dir: directory where the tiles are stored
        input_image: path to the input image to be "mosaicified"
        target_image: name of the mosaic iamage (default= "mosaic-" + input_image name)
        tilesize: size of tiles. Resized if needed
        mintiles: minimum number of tiles in a given direction. The size of the mosaic image is computed accordingly
        """

        self._tiles_dir = tiles_dir
        self._input_image_filename = input_image
        self._input_image = np.array(Image.open(input_image).convert('RGB'))
        self._tilesize = tilesize
        self._mintiles = mintiles
        self.use_luminance = use_luminance
        print("Input image size:", self._input_image.shape)
        print(f"Min number of tiles:{self._mintiles} of size {self._tilesize}")

        # Get tiles (R,G,B) average values for each image in the tile directory
        self._bucket_size = DEF_BUCKET_SIZE
        self.get_tiles_stats(self._tiles_dir, method)

        self.image_stats_computed = False

    @property
    def bucket_size(self):
        return self._bucket_size

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

    @bucket_size.setter
    def bucket_size(self, bucket_size):
        self._bucket_size = bucket_size
        self._set_distance_computing_method(self._compute_dist_method)

    @input_image_filename.setter
    def input_image_filename(self, input_image):
        self._input_image_filename = input_image
        self._input_image = np.array(Image.open(input_image).convert('RGB'))
        self.image_stats_computed = False

    @tilesize.setter
    def tilesize(self, tilesize):
        self._tilesize = tilesize
        self.image_stats_computed = False

    @mintiles.setter
    def mintiles(self, mintiles):
        self._mintiles = mintiles
        self.image_stats_computed = False

    @tiles_dir.setter
    def tiles_dir(self, tiles_dir):
        self._tiles_dir = tiles_dir
        self.get_tiles_stats(self._tiles_dir)

    def compute_image_stats(self, save=False):

        r = self._tilesize[0] / self._tilesize[1]

        # Compute the pooling size in pixels to get at least mintiles number of tiles in each directions
        sx = self._input_image.shape[0] // self._mintiles
        sy = sx // r

        sy_bis = self._input_image.shape[1] // self._mintiles
        sx_bis = int(np.around(sy_bis * r))

        # keep the lowest
        arr = np.array([[sx, sy], [sx_bis, sy_bis]]).astype(int)
        index = np.argmin(arr.sum(axis=0))

        # Get the number of tiles
        self.tiles = (self._input_image.shape[0] // arr[index, 0],
                      self._input_image.shape[1] // arr[index, 1])

        print(f"Computing average RGB values of input image in {self.tiles[0]}x{self.tiles[1]} = {np.prod(self.tiles)} boxes")

        # Compute the tile size in the input_image
        self._input_image_tilesize = (self._input_image.shape[0] // self.tiles[0],
                                      self._input_image.shape[1] // self.tiles[1])

        # Average pooling using tilesize -> each pixel store the R,G,B average in each tile.
        self.pooled_image = average_pooling(self._input_image, self.tiles,
                                            self._input_image_tilesize)

        self.image_stats_computed = True

    def get_tiles_stats(self, tiles_dir, method='brute-force'):
        """Get the RGB data of the tileset
        If RGBdata.json does not exist in the tiles directory, it is computed
        """

        if not os.path.exists(os.path.join(tiles_dir, 'RGBdata.json')):
            print("Computing RGB stats of tile images")
            create_RGB_stats(tiles_dir)

        with open(os.path.join(tiles_dir, 'RGBdata.json'), 'r') as f:
            print("Loading RGBdata.json")
            self.tilesdata = json.load(f)

        self._set_distance_computing_method(method)

    def _set_distance_computing_method(self, method):

        if method not in ["brute-force", "kdtree"]:
            print("Method to minimize RGB distance must be either {} or {}".format("brute-force", "kdtree"))
            print("Set it to default (brute-force)")
            self._compute_dist_method = 'brute-force'
        else:
            self._compute_dist_method = method

        if self._compute_dist_method == 'brute-force':
            # self.keylist = nb.typed.List(self.tilesdata.keys())
            # Shuffle images list and data
            temp = list(zip(list(self.tilesdata.keys()), list(self.tilesdata.values())))
            shuffle(temp)
            allkeys, self.RGBarray = zip(*temp)
            self.RGBarray = np.array(self.RGBarray)
            allkeys = nb.typed.List(allkeys)
            # Creating n buckets of "bucket_size" images
            ntot = len(allkeys)
            n = int(np.ceil(len(allkeys) / self._bucket_size))
            print("Creating {} buckets of {} tiles".format(n, min(self._bucket_size, self.RGBarray.size)))
            self.keylist = [allkeys[i * self._bucket_size:min((i + 1) * self._bucket_size, ntot)] for i in range(n)]
            self.RGBarray = [
                self.RGBarray[i * self._bucket_size: min((i + 1) * self._bucket_size, ntot)] for i in range(n)]

        elif self._compute_dist_method == 'kdtree':
            self.RGBarray = np.array(list(self.tilesdata.values()))
            self.keylist = list(self.tilesdata.keys())
            self.kdtree = cKDTree(list(self.tilesdata.values()))

    def build_mosaic(self, filename=None, reuse=0, randomize=True, opacity=0,
                     method="brute-force", bucket_size=None, kdtree_eps=EPS,
                     use_luminance=None):

        if method != self._compute_dist_method or (bucket_size is not None and bucket_size != self._bucket_size):
            self._bucket_size = bucket_size
            self._set_distance_computing_method(method)

        if use_luminance is not None:
            self.use_luminance = use_luminance

        # Compute average (R,G,B) values in boxes in original image if needed
        if not self.image_stats_computed:
            self.compute_image_stats()

        temp_RGBdata = deepcopy(self.RGBarray)

        mosaic_shape = (self.tiles[0] * self._tilesize[0],
                        self.tiles[1] * self._tilesize[1],
                        3)
        print("Using {} method".format(self._compute_dist_method))
        print("Mosaic will be made of {}x{} = {} tiles".format(self.tiles[0], self.tiles[1], np.prod(self.tiles)))
        print("Its size in pixels will be {}x{}".format(mosaic_shape[0], mosaic_shape[1]))

        self.mosaic_image = np.zeros(mosaic_shape).astype(np.uint8)

        if np.prod(self.tiles) > len(self.tilesdata):
            print("Not enough tiles to make mosaic without reuse")

        totsize = np.prod(self.tiles)

        if reuse < len(self.tilesdata.values()):
            tilescounter = np.zeros(len(self.tilesdata.values()))

        # Assemble tiles
        k = 0
        h, w = self.pooled_image.shape[:-1]
        # Iterate randomly so that best tile matches are not always at the same place
        if randomize:
            randomRange = np.arange(h * w)
            np.random.shuffle(randomRange)
        else:
            randomRange = range(h * w)

        # TODO: this can be parallelized with one bucket assigned to a set of the random coords
        for n in randomRange:
            i, j = divmod(n, w)
            k += 1
            print("Assembling tiles: {:04.1f}%".format(100 * k / totsize), flush=True, end='\r')

            if self._compute_dist_method == 'brute-force':
                # key = _get_best_match(self.pooled_image[i,j,:],self.keylist,temp_RGBdata)
                bucket = np.random.randint(0, len(self.keylist))
                if self.use_luminance:
                    key = _get_best_match_with_luminance(self.pooled_image[i, j, :], self.keylist[bucket], temp_RGBdata[bucket])
                else:
                    key = _get_best_match(self.pooled_image[i, j, :], self.keylist[bucket], temp_RGBdata[bucket])

                if reuse < len(self.tilesdata):
                    # ind = self.keylist.index(key)
                    ind = self.keylist[bucket].index(key)
                    tilescounter[ind] += 1
                    if tilescounter[ind] > reuse:
                        # Deleting the item in tilesdata is too slow. Just add a big value to RGB values so this image won't be chosen anymore
                        # self.RGBarray[ind,:] += 2048
                        temp_RGBdata[bucket][ind, :] += 2048

            elif self._compute_dist_method == 'kdtree':
                index = self.kdtree.query(self.pooled_image[i, j, :], k=reuse, eps=kdtree_eps)
                if reuse > 1:
                    key = self.keylist[index[1][np.random.randint(0, reuse)]]
                else:
                    key = self.keylist[index[1]]
            try:
                t = np.array(Image.open(os.path.join(self._tiles_dir, key)))
                if t.ndim == 2:
                    self.mosaic_image[i * self._tilesize[0]:(i + 1) * self._tilesize[0],
                                    j * self._tilesize[1]:(j + 1) * self._tilesize[1],
                                    :] = t[..., np.newaxis]
                else:
                    self.mosaic_image[i * self._tilesize[0]:(i + 1) * self._tilesize[0],
                                    j * self._tilesize[1]:(j + 1) * self._tilesize[1],
                                    :] = t
            except Exception as e:
                print(
                    "Problem loading and inserting tile {}. \n Check that the images in tiles directory are all of size {}".format(
                        key,
                        self._tilesize))
                print(e)

        # Saving mosaic
        if filename is None:
            filename = "mosaic-reuse{}-{}x{}-TS{}x{}-{}-o{}-B{}-L{}-{}".format(
                reuse,
                self.tiles[0],
                self.tiles[1],
                self._tilesize[0],
                self._tilesize[1],
                method,
                opacity,
                len(self.RGBarray),
                int(self.use_luminance),
                os.path.basename(self._input_image_filename))

        print("Saving Mosaic...", end="\r")

        if opacity > 0:
            w, h = self.mosaic_image.shape[:2]
            op_im = Image.fromarray(self._input_image).resize((h, w), resample=2, reducing_gap=3.0)
            blended_image = np.array(op_im).astype(float) * opacity + self.mosaic_image.astype(float) * (1 - opacity)
            PILim = Image.fromarray(np.around(blended_image).astype(np.uint8))
            PILim.save(filename)
        else:
            PILim = Image.fromarray(self.mosaic_image)
            PILim.save(filename)

        print("Mosaic saved in ", filename)

        return self.mosaic_image
