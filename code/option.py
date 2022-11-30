import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

# Train specifications
parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs to train')

parser.add_argument('--input_size', type=int, default=256, help='input size')    

parser.add_argument('--batch_size', type=int, default=8, help='batch size')                             

parser.add_argument('--is_noise', type=bool, default=True, help='noise on/off')   

parser.add_argument('--noise_value', type=float, default=1.0,
                    help='noise scale')                          

parser.add_argument('--is_log_scale', type=bool, default=True,
                    help='log scale on/off')      

parser.add_argument('--is_origin_size', type=bool, default=False,
                    help='True if no super-resolution') 

parser.add_argument('--is_hdr', type=bool, default=True,
                    help='True if hdr')           

parser.add_argument('--is_resume', type=bool, default=False,
                    help='resume on/off')    

parser.add_argument('--is_train', type=str2bool, default=True,
                    help='train on/off')     

# Ablation
parser.add_argument('--demosaicing', type=str2bool, default=True,
                    help='demosaicing on/off')         

parser.add_argument('--multiscale', type=str2bool, default=True,
                    help='multiscale on/off')     

parser.add_argument('--hdrfusion', type=str2bool, default=True,
                    help='hdrfusion on/off')

parser.add_argument('--transformer', type=str2bool, default=True,
                    help='transformer on/off')    

# Comparison
parser.add_argument('--comparison', action='store_true', default=False,
                        help='keep_query')          

# Burst
parser.add_argument('--burst', action='store_true', default=False,
                        help='keep_query')   

# Note                  
parser.add_argument('--name', type=str, default='noname',
                    help='note for model') 
         
parser.add_argument('--result_dir', type=str, default='base_dir',
                    help='dir for model')                  

# Visualization
parser.add_argument('--print_every', type=int, default=100,
                    help='number of epochs to print') 

args = parser.parse_args()