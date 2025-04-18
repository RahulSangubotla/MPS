#from https://github.com/victoresque/pytorch-template with heavy modifications
import torch
import os
import sys
import signal
import json
import logging
import argparse
from model import *
from model.loss import *
from model.metric import *
from data_loader import getDataLoader
from trainer import *
from logger import Logger

torch.serialization.add_safe_globals([Logger])
logging.basicConfig(level=logging.INFO, format='')
def set_procname(newname):
        from ctypes import cdll, byref, create_string_buffer
        newname=os.fsencode(newname)
        libc = cdll.LoadLibrary('libc.so.6')    
        buff = create_string_buffer(len(newname)+1) 
        buff.value = newname                 
        libc.prctl(15, byref(buff), 0, 0, 0) 

def main(config, resume):
    supercomputer = config.get('super_computer', False)
    train_logger = Logger()

    split = config.get('split', 'train')
    data_loader, valid_data_loader = getDataLoader(config, split)

    model = eval(config['arch'])(config['model'])
    if 'style' in config['model'] and 'lookup' in config['model']['style']:
        model.style_extractor.add_authors(data_loader.dataset.authors)
    # model.summary()
    
    if config['trainer']['class'] == 'HWRWithSynthTrainer':
        gen_model = model
        model = model.hwr
        gen_model.hwr = None
    
    loss = {name: eval(l) for name, l in config['loss'].items()} if isinstance(config['loss'], dict) else eval(config['loss'])
    
    metrics = {name: [eval(metric) for metric in m] for name, m in config['metrics'].items()} if isinstance(config['metrics'], dict) else [eval(metric) for metric in config['metrics']]
    
    trainerClass = eval(config['trainer'].get('class', 'Trainer'))
    trainer = trainerClass(model, loss, metrics,
                           resume=resume,
                           config=config,
                           data_loader=data_loader,
                           valid_data_loader=valid_data_loader,
                           train_logger=train_logger)
    
    if config['trainer']['class'] == 'HWRWithSynthTrainer':
        trainer.gen = gen_model

    def handleSIGINT(sig, frame):
        trainer.save()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handleSIGINT)
    print("Begin training")
    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--soft_resume', default=None, type=str, help='path to checkpoint that may or may not exist (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int, help='gpu to use (overrides config) (default: None)')

    args = parser.parse_args()
    config = None
    
    if args.config is not None:
        config = json.load(open(args.config))
    
    if args.resume is None and args.soft_resume is not None:
        if not os.path.exists(args.soft_resume):
            print(f'WARNING: resume path ({args.soft_resume}) was not found, starting from scratch')
        else:
            args.resume = args.soft_resume
    elif args.resume is not None and (config is None or not config.get('override', False)):
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume, map_location=torch.device('cuda'))['config']
    elif args.config is not None and args.resume is None:
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if os.path.exists(path):
            if any('checkpoint' in file for file in os.listdir(path)):
                raise Exception(f'ERROR: Path {path} already used!')
    
    assert config is not None
    name = config['name']
    file_name = args.config.split('/')[-1][:-5]
    
    if name != file_name:
        raise Exception(f'ERROR: name and file name do not match, {name} != {file_name} ({args.config})')
    
    if args.gpu is not None:
        config['gpu'] = args.gpu
        print(f'override gpu to {config["gpu"]}')
    
    if config['cuda']:
        with torch.cuda.device(config['gpu']):
            main(config, args.resume)
    else:
        main(config, args.resume)
