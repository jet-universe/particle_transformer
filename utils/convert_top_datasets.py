import os
import pandas as pd
import numpy as np
import awkward0
from uproot3_methods import TLorentzVectorArray
import awkward as ak
import argparse


'''
Datasets introduction:
  - The Machine Learning landscape of top taggers: 
    - https://scipost.org/SciPostPhys.7.1.014

Download:
  - https://zenodo.org/record/2603256
'''


def _transform(dataframe, start=0, stop=-1):

    df = dataframe.iloc[start:stop]

    def _col_list(prefix, max_particles=200):
        return ['%s_%d' % (prefix, i) for i in range(max_particles)]

    _px = df[_col_list('PX')].values
    _py = df[_col_list('PY')].values
    _pz = df[_col_list('PZ')].values
    _e = df[_col_list('E')].values

    mask = _e > 0
    n_particles = np.sum(mask, axis=1)

    px = awkward0.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward0.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward0.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward0.JaggedArray.fromcounts(n_particles, _e[mask])

    p4 = TLorentzVectorArray.from_cartesian(px, py, pz, energy)

    jet_p4 = p4.sum()

    # outputs
    v = {}
    v['label'] = df['is_signal_new'].values

    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_energy'] = jet_p4.energy
    v['jet_mass'] = jet_p4.mass
    v['jet_nparticles'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign == 0] = 1
    v['part_deta'] = (p4.eta - v['jet_eta']) * _jet_etasign
    v['part_dphi'] = p4.delta_phi(jet_p4)

    return v


def convert(source, destdir, basename):
    df = pd.read_hdf(source, key='table')
    print('Total events: %s' % str(df.shape[0]))
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    output = os.path.join(destdir, '%s.parquet' % basename)
    print(output)
    if os.path.exists(output):
        os.remove(output)
    v = _transform(df)
    arr = ak.Array({k: ak.from_awkward0(a) for k, a in v.items()})
    ak.to_parquet(arr, output, compression='LZ4', compression_level=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert top benchmark h5 datasets')
    parser.add_argument('-i', '--inputdir', required=True, help='Directory of input h5 files.')
    parser.add_argument('-o', '--outputdir', required=True, help='Output directory.')
    args = parser.parse_args()

    # conver training file
    convert(os.path.join(args.inputdir, 'train.h5'), destdir=args.outputdir, basename='train_file')

    # conver validation file
    convert(os.path.join(args.inputdir, 'val.h5'), destdir=args.outputdir, basename='val_file')

    # conver testing file
    convert(os.path.join(args.inputdir, 'test.h5'), destdir=args.outputdir, basename='test_file')
