# This is the CLI for Beholder

from argparse import ArgumentParser
from src import image, util, vqgan
import os
import datetime
import yaml

all_args = ArgumentParser(description='Interpret source images.')
all_args.add_argument('-i', '--input', dest='input', type=str, required=True, help='Path to the folder(s) of images to interpret.')
all_args.add_argument('-o', '--output', dest='output', type=str, required=True, help='Path to the folder of output images.')
all_args.add_argument('-n', '--iterations', dest='iterations', action='append', type=int, required=True, help='Number of iterations to run. If providing multiple, be sure to include an interpolation mark in a prefix or suffix: {}')
all_args.add_argument('-c', '--cut-frame', dest='cut_frames', action='append', type=int, required=False, help='Index of frame which represent new shot/cut.')
all_args.add_argument('-C', '--cut-threshhold', dest='cut_threshold', type=int, required=False, help='Alternate to providing cut frame indices, this is a percentage threshold of image difference to consider a cut.')
all_args.add_argument('-N', '--cut-iterations', dest='cut_iterations', type=int, required=False, help='Extra iterations to run on any frame in a new shot/cut (includes starting frame).')
all_args.add_argument('-p', '--prefix', dest='prefix', type=str, required=False, help='Prefix to add for output image filenames.')
all_args.add_argument('-s', '--suffix', dest='suffix', type=str, required=False, help='Suffix to add for output image filenames.')
all_args.add_argument('-t', '--tween', dest='tween', type=int, required=False, help='Tween between iterations, num weighs towards destination frame.')
all_args.add_argument('-I', '--image-prompts', action='append', dest='image_prompts', type=str, required=False, help='Image prompt path for GAN.')
all_args.add_argument('-T', '--text-prompts', action='append', dest='text_prompts', type=str, required=False, help='Text prompt for GAN.')
all_args.add_argument('-f', '--force', dest='force', required=False, default=False, type=bool, help='Force overwrite of output images.')
args = all_args.parse_args()

util.header('Behold Images!')
print(yaml.dump(args))

interpreter = None
interpreter_image_size = None

last_input_image = None
last_output_image = None
tween_image = None
mount_image = None

image_prompts = args.image_prompts or []
text_prompts = args.text_prompts or []
cut_frames = args.cut_frames or []

input_images = os.listdir(args.input)

input_images = [f for f in filter(lambda f: (f[0] != '.'), os.listdir(args.input))]
input_images.sort()

total_images = len(input_images)
image_count = 0

util.logger.info('Found {} images.'.format(total_images))

process_start_time = datetime.datetime.now()

prime_iterations = args.iterations[0]

# Loop through all sorted images
for i in input_images:
    image_count += 1
    image_start_time = datetime.datetime.now()
    input_image = os.path.join(args.input, i)
    output_image = i
    [basename, ext] = os.path.splitext(output_image)

    # Add prefix to output image name
    if args.prefix is not None:
        output_image = args.prefix + output_image
        [basename, ext] = os.path.splitext(output_image)
    # Add suffix to output image name
    if args.suffix is not None:
        output_image = basename + args.suffix + ext
        [basename, ext] = os.path.splitext(output_image)
    
    # Compose full output image path
    output_image = os.path.join(args.output, output_image)

    # Skip generation of image if it already exists, unless force is set
    if os.path.exists(output_image.format(prime_iterations)) and not args.force:
        util.logger.debug(output_image.format(prime_iterations) + ' already exists, skipping.')
        last_input_image = input_image
        last_output_image = output_image.format(prime_iterations)
        continue

    input_image_size = image.get_size(input_image)

    current_frame_is_a_cut = False
    if image_count == 1:
        current_frame_is_a_cut = True
    elif image_count in cut_frames:
        current_frame_is_a_cut = True
    elif args.cut_threshold is not None and last_input_image is not None and image.get_difference(last_input_image, input_image) >= args.cut_threshold:
        current_frame_is_a_cut = True

    iterations_boost = 0
    last_input_image = input_image

    # When current frame is a cut and tweening is on we will boost the iterations for this frame by cut_iterations
    if args.tween is not None and args.cut_iterations is not None and current_frame_is_a_cut:
        iterations_boost = args.cut_iterations

    # Generate tween image and make it the input_image *unless* current frame is a cut.
    elif args.tween is not None and last_output_image is not None and not current_frame_is_a_cut:
        util.logger.debug('Tweening from {}->[{}]->{}'.format(last_output_image, args.tween, input_image))
        tween_image = util.tempfile(ext)

        # This mounting code is necessary if there's a size mismatch between the from/to images
        # for the tween.  However, it's not clear that this is the best way to do this as it
        # may result in drifting artifacts from the tween when used as a video.
        if image.get_size(last_output_image) != input_image_size:
            image.mount(input_image, tween_image, image.get_size(last_output_image))
            input_image = tween_image

        # Perform the tween, producing a tween image.
        image.tween(last_output_image, input_image, tween_image, args.tween)
        tween_image_size = image.get_size(tween_image)
        if tween_image_size != input_image_size:
            image.mount(tween_image, tween_image, input_image_size)
        input_image = tween_image

    # We need a new interpreter when the image size changes
    if not interpreter or interpreter_image_size != input_image_size:
        interpreter_image_size = input_image_size
        interpreter = vqgan.generate_interpreter(interpreter_image_size, text_prompts, image_prompts)

    # Hallucination time!
    interpreter(input_image, output_image, args.iterations, iterations_boost)

    # Fix the output image(s) size if necessary.  This also may not be the best
    # way to do this.
    if image.get_size(output_image.format(prime_iterations)) != input_image_size:
        for i in args.iterations:
            image.resize(output_image.format(i), output_image.format(i), input_image_size)

    last_output_image = output_image.format(prime_iterations)
    image_end_time = datetime.datetime.now()
    for i in args.iterations:
        util.logger.info('Generated image {} of {} in {}: {}'.format(image_count, total_images, str(image_end_time - image_start_time), output_image.format(i)))

    # Clean up temp files
    if not tween_image is None:
        tween_image = None
        util.rm(tween_image)

process_end_time = datetime.datetime.now()

util.logger.info('Completed after {}'.format(str(process_end_time - process_start_time)))