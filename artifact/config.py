params = {}#定义超参数

params['train_path'] = 'G:\CT_DATA'
params['batch_size'] = 16
params['num_workers'] = 4
params['lr'] = 1e-6
params['weight_decay'] = 1e-5
params['save_path'] = './'
params['log'] = './log/'
params['gpu'] = [0]
params['pretrained'] = './2'
params['epoch_num'] = 10
params['test_only'] = False
