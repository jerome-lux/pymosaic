import argparse
from pymosaic import MosaicMaker, create_tileset

parser = argparse.ArgumentParser(
    description='Creates resized copy of n images in input_directory to outpout_directory ',
    fromfile_prefix_chars='@')
parser.add_argument("-i", "--input", dest="input_dir",
                    help="input directory")
parser.add_argument("-o", "--output", dest="output_dir",
                    help="output directory.")
parser.add_argument("--max", dest="max",default=None,
                    help="Maximum number of images to resize")
parser.add_argument("--size", dest="size", nargs=2, default=(64, 64),
                    help="Size of tiles in pixels: nx, ny")
parser.add_argument("--crop_aspect_ratio", dest="crop_aspect_ratio", , default=None
                    help="Image are cropped to respect the aspect ratio")

parser.parse_args()

args = parser.parse_args()

create_tileset(args.input,args.output,size=args.size,max_im = args.max, crop_aspect_ratio=args.crop_aspect_ratio)