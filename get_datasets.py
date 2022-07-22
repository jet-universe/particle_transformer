#!/usr/bin/env python3

import argparse
import os
import shutil
from utils.dataset_utils import get_file, extract_archive


datasets = {
    'JetClass': {
        'Pythia/train_100M': [
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part0.tar', 'de4fd2dca2e68ab3c85d5cfd3bcc65c3'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part1.tar', '9722a359c5ef697bea0fbf79bf50f003'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part2.tar', '1e9f66cd1f915f9d10e90ae1d7761720'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part3.tar', '47348fc8985319fa4806da87500482fa'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part4.tar', '6b0ce16bd93b442a8d51914466990279'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part5.tar', '416e347512e716de51d392bee327b8e9'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part6.tar', 'e9b9c1557b1b39bf0a16e4ab631ae451'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part7.tar', '5bfc6cb285ccb7680cefa9ac82ad1a2e'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part8.tar', '540c1a0d66dfad78d2b363c5740ccf86'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part9.tar', '668f40b3275167ff7104c48317c0ae2a'),
        ],
        'Pythia/': [
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_val_5M.tar', '7235ccb577ed85023ea3ab4d5e6160cf'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_test_20M.tar', '64e5156d26d101adeb43b8388207d767'),
        ],
    },
    'TopLandscape': {
        # converted from https://zenodo.org/record/2603256
        '../': [
            ('https://hqu.web.cern.ch/datasets/TopLandscape/TopLandscape.tar', '4fca2e47afbf321b0f201da6b804c404'),
        ],
    },
    'QuarkGluon': {
        # converted from https://zenodo.org/record/3164691
        '../': [
            ('https://hqu.web.cern.ch/datasets/QuarkGluon/QuarkGluon.tar', 'd8dd7f71a7aaaf9f1d2ee3cddef998f9'),
        ],
    },
}


def download_dataset(dataset, basedir, envfile, force_download):
    info = datasets[dataset]
    datadir = os.path.join(basedir, dataset)
    if force_download:
        if os.path.exists(datadir):
            print(f'Removing existing dir {datadir}')
            shutil.rmtree(datadir)
    for subdir, flist in info.items():
        for url, md5 in flist:
            fpath, download = get_file(url, datadir=datadir, file_hash=md5, force_download=force_download)
            if download:
                extract_archive(fpath, path=os.path.join(datadir, subdir))

    datapath = f'DATADIR_{dataset}={datadir}'
    with open(envfile) as f:
        lines = f.readlines()
    with open(envfile, 'w') as f:
        for l in lines:
            if f'DATADIR_{dataset}' in l:
                l = f'export {datapath}\n'
            f.write(l)
    print(f'Updated dataset path in {envfile} to "{datapath}".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=datasets.keys(), help='datasets to download')
    parser.add_argument('-d', '--basedir', default='datasets', help='base directory for the datasets')
    parser.add_argument('-e', '--envfile', default='env.sh', help='env file with the dataset paths')
    parser.add_argument('-f', '--force', action='store_true', help='force to re-download dataset')
    args = parser.parse_args()

    download_dataset(args.dataset, args.basedir, args.envfile, args.force)
