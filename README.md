# Pymosaic

A Python module to create photomosaic images.  
This implementation use numba to speed up the mosaic building, but it can be slow when the number of tiles in the image pool is large.
Currently, there is no transparency. Tiles are chosen based the distance (in RGB space) minimization between the average RGB values in a tile and the average RGB values in each box in target image.

To create a photomosaic, you just need:
- An target image
- A pool of images with same shapes

## 1. Create a pool of resized images:

    Use the function create_tileset(input_dir, target_dir, size=(64, 64), max_im=None, replace=False, ncpu=cpu_count())
    - input_dir: directory where images are stored (recursive search, so all images in subfolders are processed).
    - target_dir: where the resized images are saved
    - size: (tuple) (nx,ny) image size
    - max_im: maximum number of processed images
    - replace: if True, rdelete the target_directory, else, just add images.
    
    This function also creates a RGBdata.json that stores the average R,G and B values for each image in the target directory

## 2. Create a MosaicMaker object:

    mosaicmaker = MosaicMaker(input_image="path/to/target/image",tiles_dir="path/to/images/pool/",mintiles120,tilesize=(50,50))
    where:
    - input_image: target image
    - tiles_dir: directory containing images used as tiles (must be of the same size)
    - mintiles: minimum number of tiles horizontally or vertically
    - tilessize: tuple giving the size of the tiles (must be the same as the images in tiles_dir)
    
## 3. Build the mosaic :

    mosaic_image = mosaicmaker.build_mosaic(filename="path/to/mosaic/image.jpg,reuse=20,randomize=True)
    where:
    - filename is the name of the mosaic that will be created.  
      If None, it creates a file in current directory using the input_filename:  
      "mosaic-reuse{XX}-{tilesinX}x{tilesinY}-TS{tilesizeX}x{}-{tilesizeY}" + input_image
    - reuse: number of time a tile can be reused in the mosaic. Higher values creates better mosaic, but with less variety. Very low value could produce strange results... If the number of images in the tiles_dir is not sufficient, tiles will be reused (after all images have been used 'reused' times)
    - randomize: iterate the image randomly. Way better that sequentially, especially if the number of tile is small.  
    
    build_mosaic also return the created mosaic image

## 4. Tips
- Try different reuse values to tune the mosaic. With small reuse values, the produced mosaic could be less accurate, but it can creates interesting effects. High reuse values produce accurate mosaic, but with no tile variety in homogeneous zones.
- Use max_im to limit the number of tiles in the tiles directory, as the computation time increases with the number of tiles.
