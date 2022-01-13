from src import rife, util

def blend(img_1, img_2, amount, out_path):
    util.cmd([
        'composite', img_1, img_2,
        '-blend', amount,
        '-gravity', 'center',
        out_path])

def brighten(in_path, amount, out_path):
    util.cmd([
        'convert', in_path,
        '-modulate', '{a}%'.format(a=amount),
        out_path])

def contrast(in_path, amount, out_path):
    util.cmd([
        'convert', in_path,
        *(['-contrast' if amount > 0 else '+contrast'] * abs(amount)),
        out_path])

def get_size(image_path):
    [w, h] = util.cmd([
        'identify',
        '-format', '%w,%h',
        image_path
    ]).split(',')
    return int(w), int(h)

def mount(in_path, out_path, size):
    util.cmd([
        'convert', in_path,
        '-background', 'black',
        '-gravity', 'center',
        '-extent', '{x}x{y}'.format(x=size[0], y=size[1]),
        out_path])

def sharpen(in_path, out_path, amount):
    util.cmd([
        'convert',
        '-sharpen', amount,
        in_path, out_path])

def tween(from_path, to_path, out_path, exp, index):
    rife.inference_img(
      img0_path=from_path,
      img1_path=to_path,
      exp=exp,
      out_path=out_path)

def zoom(in_path, out_path, size, amount):
    resize_arg = '{x}x{y}^'.format(x=size[0]+amount, y=size[1]+amount)
    if amount < 0:
        util.cmd([
            'convert', in_path, out_path,
            '-resize', resize_arg])
        util.cmd([
            'convert', in_path, out_path,
            '-gravity', 'center',
            '-composite', out_path])
    if amount > 0:
        crop_arg = '{x}x{y}+{o}+{o}'.format(x=size[0], y=size[1], o=int(amount/2))
        util.cmd([
            'convert', in_path, out_path,
            '-resize', resize_arg,
            '-crop', crop_arg])
    if amount == 0:
        util.copy_file(in_path, out_path)
