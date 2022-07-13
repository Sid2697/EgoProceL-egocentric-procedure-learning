"""
This file is use to crawl and download the CMU kitchens dataset
Link to the webpage: http://kitchen.cs.cmu.edu/
"""

import os
import time
import argparse
import numpy as np

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument(
    '--link',
    default='http://kitchen.cs.cmu.edu/main.php',
    help='Link to the webpage containing data'
)
parser.add_argument(
    '--dir',
    default='/Volumes/Storage/Egocentric/Datasets/CMU_Kitchens/',
    help='Path to the directory where data is to be downloaded'
)
args = parser.parse_args()


def unzip_del(zip_file):
    unzip_command = 'unzip {} -d {}'
    delete_command = 'rm {}'
    dest_dir = zip_file.split('.')[0]
    if not os.path.isdir(dest_dir):
        print('[INFO] Creating {}...'.format(dest_dir))
        os.mkdir(dest_dir)
        print('[INFO] Unzipping {}'.format(zip_file))
        os.system(unzip_command.format(zip_file, dest_dir))
        print('[INFO] Deleting {}...'.format(zip_file))
        os.system(delete_command.format(zip_file))
    else:
        print('[INFO] Files for {} already extracted...'.format(zip_file))
    return None


req = Request(args.link)
html_page = urlopen(req)

soup = BeautifulSoup(html_page, "lxml")

links = []
for link in soup.findAll('a'):
    links.append(link.get('href'))

download_link = 'http://kitchen.cs.cmu.edu/{}'
download_command = 'wget -P {} {}'

for link in links:
    if link is not None:
        if '.zip' in link and 'Video' in link:
            activity = link.split('_')[1]
            activity_dir = os.path.join(args.dir, activity)
            if os.path.isdir(activity_dir):
                pass
            else:
                os.mkdir(activity_dir)
            file_path = os.path.join(activity_dir, link.split('/')[-1]).split(
                '.'
            )[0]
            if os.path.isdir(file_path):
                print('[INFO] {} exists...'.format(link))
                pass
            else:
                print('[INFO] Downloading {}...'.format(link))
                download_link_ = download_link.format(link)
                download_command_ = download_command.format(
                    activity_dir,
                    download_link_
                )
                os.system(download_command_)
                unzip_del(file_path)
                time.sleep(np.random.randint(2, 15))
