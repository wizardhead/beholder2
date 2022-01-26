import logging
import os
from os.path import abspath, dirname, exists, join
import pyfiglet
from subprocess import PIPE, run
from tempfile import NamedTemporaryFile, gettempdir
import uuid

HEADER_FIGLET = pyfiglet.Figlet()

def cmd(args, cwd=None):
    global logger
    args = list(map(lambda x: str(x), args))
    logger.debug("$ " + " ".join(args).strip())
    result = system_result(args, cwd).strip()
    if result != '':
        logger.debug(result)
    return result

def copy_file(from_path, to_path):
    cmd(['cp', from_path, to_path])

def config_logger(loglevel=None):
    loglevel = dict(
        debug=logging.DEBUG,
        info=logging.INFO,
        warn=logging.WARN,
        error=logging.ERROR,
    ).get(loglevel)
    logger = logging.getLogger()
    logHandler = logging.StreamHandler()

    if not loglevel is None:
        logger.setLevel(loglevel)
        logHandler.setLevel(logging.DEBUG)

    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logHandler.setFormatter(logFormatter)

    logger.addHandler(logHandler)
    return logger

def header(text):
    global logger
    logger.info(HEADER_FIGLET.renderText(text))

def init_logger(loglevel=None):
    global logger
    logger = config_logger(loglevel)
    return logger

def mkdirp(path):
    cmd(['mkdir', '-p', path])

def read_file(path):
    file = open(path, mode='r')
    content = file.read()
    file.close()
    return content

def report_errors(errors):
    if len(errors) == 0:
        return
    for error in errors:
        logger.error(error)
    exit(1)

def rm(path):
    os.unlink(path)

def rmrf(path):
    cmd(['rm', '-rf', path])

def relative_to_config_path(config_path, target_path):
    if target_path is None:
        return None
    return join(dirname(abspath(config_path)), target_path)

def system_result(args, cwd=None):
    if cwd is None:
        return run(args, stdout=PIPE).stdout.decode('utf-8')
    else:
        return run(args, stdout=PIPE, cwd=cwd).stdout.decode('utf-8')

def tempfile(ext):
    file = NamedTemporaryFile(suffix=ext)
    name = file.name
    file.close()
    return name
    
def tempfolder():
    folder = join(gettempdir(), 'beholder_workspaces', str(uuid.uuid4()))
    if exists(folder):
        raise Exception('tried to create temp dir, but already exists: ' + folder)
    mkdirp(folder)
    return folder

def write_file(path, content):
    file = open(path, mode='w')
    file.write(content)
    file.close()

class NonInitializedLogger:
    def print(self, *args):
        print(*args)
    def debug(self, *args):
        self.print(*args)
    def info(self, *args):
        self.print(*args)
    def warn(self, *args):
        self.print(*args)
    def error(self, *args):
        self.print(*args)
        
logger = NonInitializedLogger()
