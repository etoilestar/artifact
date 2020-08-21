params = {}#定义超参数

params['train_path'] = r'G:\1'#'G:\CT_DATA\202001\钙化伪影'
params['batch_size'] = 2
params['num_workers'] = 4
params['lr_g'] = 1e-4
params['lr_d'] = 1e-4
params['weight_decay'] = 1e-5
params['save_path'] = './'
params['structure'] = 'cyclegan'
params['log'] = './log/'
params['gpu'] = [0]
params['pretrained'] = None#'2020-08-20-20-28-00'#
params['epoch_num'] = 100
params['test_only'] = False
params['svimg_path'] = None#'train_img'

