import os
import argparse
import math
from subprocess import PIPE, run

all_args = argparse.ArgumentParser()
all_args.add_argument('-i', '--input', help='Input file', required=True)
all_args.add_argument('-o', '--output', help='Output file', required=True)

args = vars(all_args.parse_args())

input_file = args['input']
output_file = args['output']

def cmd(args, cwd=None):
  if cwd is None:
    return run(args, stdout=PIPE).stdout.decode('utf-8')
  else:
    return run(args, stdout=PIPE, cwd=cwd).stdout.decode('utf-8')

def get_slices(input_file):
  input_folder = os.path.dirname(input_file)
  [input_file_noext, input_file_ext] = os.path.splitext(os.path.basename(input_file))
  all_files = os.listdir(input_folder)
  slices = []
  for file in all_files:
    if str.startswith(file, input_file_noext + '.'):
      slices.append(input_folder + '/' + file)
  slices.sort(key=lambda f: (f.split('.')[-2]+f.split('.')[-3]))
  return slices

def montage(slices, output_file):
  last_slice = slices[-1]
  [columns, rows] = last_slice.split('.')[-3:-1]
  columns = int(columns)+1
  rows = int(rows)+1
  command = ['montage']
  for slice in slices:
    command.append(slice)
  command.append('-geometry')
  command.append('+0+0')
  command.append('-tile')
  command.append('{}x{}'.format(columns, rows))
  command.append(output_file)
  cmd(command)

montage(get_slices(input_file), output_file)
