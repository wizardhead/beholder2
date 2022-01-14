# This is the CLI for Beholder

from argparse import ArgumentParser
from src import image, util, vqgan
import os

all_args = ArgumentParser(description='Interpret source images.')
all_args.add_argument('-i', '--input', dest='input', type=str, required=True, help='Path to the folder of images to interpret.')
all_args.add_argument('-o', '--output', dest='output', type=str, required=True, help='Path to the folder of output images.')
all_args.add_argument('-n', '--iterations', dest='iterations', type=int, required=True, help='Number of iterations to run.')
all_args.add_argument('-p', '--prefix', dest='prefix', type=str, required=False, help='Prefix to add for output image filenames.')
all_args.add_argument('-s', '--suffix', dest='suffix', type=str, required=False, help='Suffix to add for output image filenames.')
all_args.add_argument('-I', '--image-prompts', action='append', dest='image_prompts', type=str, required=False, help='Image prompt path for GAN.')
all_args.add_argument('-T', '--text-prompts', action='append', dest='text_prompts', type=str, required=False, help='Text prompt for GAN.')

args = all_args.parse_args()

interpreter = None
last_image_size = None

image_prompts = args.image_prompts or []
text_prompts = args.text_prompts or []

util.header('here we go!')
input_images = os.listdir(args.input)

input_images = [f for f in filter(lambda f: (f[0] != '.'), os.listdir(args.input))]
input_images.sort()

for i in input_images:
    input_image = os.path.join(args.input, i)
    output_image = i
    if args.prefix is not None:
        output_image = args.prefix + output_image
    if args.suffix is not None:
        [basename, ext] = os.path.splitext(output_image)
        output_image = basename + args.suffix + ext
    output_image = os.path.join(args.output, output_image)
    image_size = image.get_size(input_image)
    if image_size != last_image_size:
        interpreter = vqgan.generate_interpreter(image_size, text_prompts, image_prompts)
    interpreter(input_image, output_image, args.iterations)

