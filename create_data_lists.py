import argparse
from ssd_utils import create_voc_data_lists, create_ainet_data_lists
import sys

if __name__ == '__main__':
    # create_voc_data_lists(voc07_path='/media/ssd/ssd data/VOC2007',
    #                       voc12_path='/media/ssd/ssd data/VOC2012',
    #                       output_folder='./')
    # create_ainet_data_lists('/home/ywlee/Dataset/AINetworks/LicencePlate',
    #                         '/home/ywlee/Dataset/AINetworks/KoreaLicencePlate',
    #                         '/home/ywlee/Dataset/AINetworks/HumanFace')
    create_ainet_data_lists(*sys.argv[1:])
