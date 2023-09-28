# import logging
# import argparse


# def cli():
#     parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
#     parser.add_argument('--data_type', default="all",  
#                         help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
#     parser.add_argument('--epochs', default=256, type=int, 
#                         help='number of epochs to train the model , (default: 256)')
#     parser.add_argument('--root_dir', default="data", 
#                         help='folder of the dataset , (default: Data)')
#     parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false', 
#                         help='use pretrained values to initalize ResNet18 , (default: True)')
#     parser.add_argument('--test_epochs', default=10, type=int, 
#                         help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')
#     parser.add_argument('--freeze_resnet', default=20, type=int, 
#                         help='number of epochs to freeze resnet (default: 20)')
#     parser.add_argument('--lr', default=0.03, type=float, 
#                         help='learning rate (default: 0.03)')
#     parser.add_argument('--optim', default="sgd", 
#                         help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')
#     parser.add_argument('--batch_size', default=64, type=int, 
#                         help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   
#     parser.add_argument('--head_layer', default=1, type=int, 
#                         help='number of layers in the projection head (default: 1)')
#     parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union'], 
#                         help='cutpaste variant to use (dafault: "3way")')
#     parser.add_argument('--cuda', default=False, action='store_true', 
#                         help='use cuda for training (default: False)')
#     parser.add_argument('--workers', default=8, type=int, 
#                         help="number of workers to use for data loading (default:8)")
#     return parser.parse_args()


# def main(args):
    

# logging.basicConfig(
#     level=logging.INFO,  # Set the logging level to INFO or another level as needed.
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     filename='myapp.log',  # Specify the name of the log file.
#     filemode='w'  # 'w' to create a new log file each time, 'a' to append to an existing file.
# )