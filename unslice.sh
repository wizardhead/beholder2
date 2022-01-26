# Rejoin all images in $1 into images at $2.  If $1 or $2 are directories,
# they must have trailing /. This is because we want to support prefix
# concatenation as an option, like "output/rejoined-" producing 
# "output/rejoined-file.jpg"
for rejoin_image in $1*.0000.0000.jpg
do
	rejoin_image=`echo $rejoin_image|sed "s/\.[0-9]*\.[0-9]*//g"`
	rejoin_basename=`basename "$rejoin_image"`
	rejoin_dirname=`dirname "$rejoin_image"`
  echo "rejoin image: '$rejoin_image'"
  echo "rejoin dirname: '$rejoin_dirname'"
  echo "rejoin basename: '$rejoin_basename'"
	bash -c "python3 unslice.py -i \"$rejoin_image\" -o \"$2$rejoin_basename\""
done


