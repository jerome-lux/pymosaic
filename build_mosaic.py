import argparse
from pymosaic import MosaicMaker, create_tileset

parser = argparse.ArgumentParser(
    description='Script to build a photomosaic',
    fromfile_prefix_chars='@')
parser.add_argument("-i", "--input", dest="input_image",
                    help="input image path")
parser.add_argument("-o", "--output", dest="output_image", default=None,
                    help="output filename. If not given, a filename is automatically generated using the input filename and the mosaic parameters")
parser.add_argument("--tiles_dir", dest="tiles_dir",
                    help="path to a directory containing the tiles")
parser.add_argument("--method", dest="method", default="brute-force", type=str,
                    help="Method used to find the nearest neighbors in RGB space('brute-force' or 'kdtree'). Default: 'brute-force')
parser.add_argument("--reuse", dest="reuse", default=0,
                    help="number of times an image can be reused in the mosaic (only for brute-force)."
                    "\n When method = 'kdtree' and reuse=k, tiles are randomly chosen within the k-th nearest neighbors ")
parser.add_argument("--tilesize", dest="tilesize", nargs=2, default=(64, 64),
                    help="Size of tiles in pixels: nx, ny")
parser.add_argument("--mintiles", dest="mintiles", default=100, type=int,
                    help="Minimum  number of tiles in a direction in the mosaic image.")
parser.parse_args()

args = parser.parse_args()

m = MosaicMaker(input_image = args.input_image,
                tiles_dir = args.tiles_dir,
                target_image = args.output_image,
                tilesize = args.tilesize,
                mintiles = args.mintiles,
                method = args.method)
                
m.build_mosaic(reuse = args.reuse, method = args.method)
