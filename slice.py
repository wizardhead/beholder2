import os
import argparse
import math
from subprocess import PIPE, run

all_args = argparse.ArgumentParser()
all_args.add_argument('-i', '--input', help='Input file', required=True)
all_args.add_argument('-o', '--output', help='Output file', required=True)
all_args.add_argument('-w', '--width', type=int, help='Max width of slice', required=True)
all_args.add_argument('-H', '--height', type=int, help='Max height of slice', required=True)

args = vars(all_args.parse_args())

input_file = args['input']
output_file = args['output']
[output_file_noext, output_file_ext] = os.path.splitext(output_file)

def cmd(args, cwd=None):
  if cwd is None:
    return run(args, stdout=PIPE).stdout.decode('utf-8')
  else:
    return run(args, stdout=PIPE, cwd=cwd).stdout.decode('utf-8')

def get_size(file):
  [width, height] = cmd(['identify', '-format', '%w %h', file]).split()
  return [int(width), int(height)]

def crop(input, tl, br, output):
  cmd(['convert', input, '-crop', '{}x{}+{}+{}'.format(br[0] - tl[0], br[1] - tl[1], tl[0], tl[1]), output])

def thumb(input, output, width, height):
  cmd(['convert', input, '-thumbnail', '{}x{}'.format(width, height), '-background', 'black', '-gravity', 'NorthWest', '-extent', '{}x{}'.format(width, height), output])

[width, height] = get_size(args['input'])
columns = math.ceil(width / args['width'])
rows = math.ceil(height / args['height'])

for column in range(0, columns):
  for row in range(0, rows):
    output_file_slice = output_file_noext + '.{:04d}.{:04d}{}'.format(column, row, output_file_ext)
    print('Cropping ' + output_file_slice)
    crop(input_file, (column * args['width'], row * args['height']), ((column + 1) * args['width'], (row + 1) * args['height']), output_file_slice)
    thumb(output_file_slice, output_file_slice, args['width'], args['height'])
