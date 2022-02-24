import os
import argparse
from subprocess import PIPE, run

all_args = argparse.ArgumentParser()
all_args.add_argument('-i', '--input', help='Input file', required=True)
all_args.add_argument('-o', '--output', help='Output file', required=True)

args = vars(all_args.parse_args())

print("Unslicing {} -> {}".format(args['input'], args['output']))

input_file = args['input']
output_file = args['output']
output_file_basename = os.path.basename(output_file)

def cmd(args, cwd=None):
  if cwd is None:
    return run(args, stdout=PIPE).stdout.decode('utf-16')
  else:
    return run(args, stdout=PIPE, cwd=cwd).stdout.decode('utf-16')

def get_slices(input_file):
  input_folder = os.path.dirname(input_file)
  input_file_basename = os.path.basename(input_file)
  all_files = os.listdir(input_folder)
  all_files.sort()
  slices = []
  for file in all_files:
    if file == input_file_basename:
      continue
    if file == output_file_basename:
      continue
    if str.startswith(file, input_file_basename + '.'):
      slices.append(input_folder + '/' + file)
  slices.sort(key=lambda f: (f.split('.')[-2]+f.split('.')[-3]))
  return slices

def montage(slices, output_file):
  last_slice = slices[-1]
  [columns, rows] = last_slice.split('.')[-3:-1]
  print(last_slice)
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

slices = get_slices(input_file)
print(input_file, 'has', len(slices), 'slices')
montage(get_slices(input_file), output_file)
