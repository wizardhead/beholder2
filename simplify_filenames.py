import re, os, sys
folder = sys.argv[1]
filenames = os.listdir(folder)
filenames.sort()
for filename in filenames:
    new_filename = re.sub(r'[^a-zA-Z0-9.]', '_', filename)
    os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))

