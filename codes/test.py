# cd .\paper\diffusion_model\SROOE\codes\
# python test.py -opt options/test/test.yml
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import os.path as osp
import logging
import argparse

import options.options as option
import utils.util as util

from data import create_dataset, create_dataloader
from models import create_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--T_ctrl', type=float, default=1.0)
    parser.add_argument('-opt', type=str, default='options/test.yml',
                        help='Path to options YMAL file.')
    opt = parser.parse_args()
    T_ctrl_temp = opt.T_ctrl
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    opt.T_ctrl = T_ctrl_temp
    T_ctrl_str = '%03d' % (opt.T_ctrl * 100)

    opt['name'] += T_ctrl_str
    opt['path']['results_root'] += T_ctrl_str
    opt['path']['log'] += T_ctrl_str
    opt['T_ctrl'] = opt.T_ctrl

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=False)  # screen 是否输出到屏幕; tofile 是否写入文件
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))  # 输出配置

    return opt, logger


if __name__ == '__main__':
    opt, log = get_args()
    dataloaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        log.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        dataloaders.append(create_dataloader(test_set, dataset_opt))

    model = create_model(opt)
    for test_loader in dataloaders:
        test_set_name = test_loader.dataset.opt['name']
        log.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        for i, data in enumerate(test_loader):
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test(opt)
            visuals = model.get_current_visuals(need_GT=need_GT)
            sr_img = util.tensor2img(visuals['SR'])  # uint8
            cm_img = util.tensor2img(visuals['CM'])  # uint8

            # save images
            suffix = opt['suffix']
            img_name = img_name + suffix + '.png' if suffix else img_name + '.png'
            save_img_path = osp.join(dataset_dir, img_name)

            util.save_img(sr_img, save_img_path)

            # save_cm_path = os.path.join(dataset_dir, '{:s}_cmap.png'.format(img_name))
            # util.save_img(cm_img, save_cm_path)

            log.info('Fig {}/{}: {}'.format(i + 1, len(test_loader), save_img_path))
