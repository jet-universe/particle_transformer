import os
import numpy as np
import awkward as ak
import argparse


'''
Datasets introduction:
https://energyflow.network/docs/datasets/#quark-and-gluon-jets

Download:
- Pythia8 Quark and Gluon Jets for Energy Flow:
  - https://zenodo.org/record/3164691

- Herwig7.1 Quark and Gluon Jets:
  - https://zenodo.org/record/3066475

Versions:
 - awkward==2.6.4
 - vector==1.4.0
'''


def _p4_from_ptetaphim(pt, eta, phi, mass):
    import vector
    vector.register_awkward()
    return vector.zip({'pt': pt, 'eta': eta, 'phi': phi, 'mass': mass})


def _transform(X, y, start=0, stop=-1):
    # source_array: (num_data, max_num_particles, 4)
    # (pt,y,phi,pid)

    X = X[start:stop].astype(np.float32)
    y = y[start:stop]

    origPT = X[:, :, 0]
    indices = np.argsort(-origPT, axis=1)

    _pt = np.take_along_axis(X[:, :, 0], indices, axis=1)
    _eta = np.take_along_axis(X[:, :, 1], indices, axis=1)
    _phi = np.take_along_axis(X[:, :, 2], indices, axis=1)
    _pid = np.take_along_axis(X[:, :, 3], indices, axis=1)

    mask = _pt > 0
    n_particles = np.sum(mask, axis=1)

    pt = ak.unflatten(_pt[mask], n_particles)
    eta = ak.unflatten(_eta[mask], n_particles)
    phi = ak.unflatten(_phi[mask], n_particles)
    mass = ak.zeros_like(pt)
    PID = ak.unflatten(_pid[mask], n_particles)

    p4 = _p4_from_ptetaphim(pt, eta, phi, mass)
    px = p4.x
    py = p4.y
    pz = p4.z
    energy = p4.energy

    jet_p4 = ak.sum(p4, axis=1)

    # outputs
    v = {}
    v['label'] = y

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

    _jet_etasign = ak.to_numpy(np.sign(v['jet_eta']))
    _jet_etasign[_jet_etasign == 0] = 1
    v['part_deta'] = (p4.eta - v['jet_eta']) * _jet_etasign
    v['part_dphi'] = p4.deltaphi(jet_p4)

    v['part_pid'] = PID
    v['part_isCHPlus'] = ak.values_astype((PID == 211) + (PID == 321) + (PID == 2212), 'float32')
    v['part_isCHMinus'] = ak.values_astype((PID == -211) + (PID == -321) + (PID == -2212), 'float32')
    v['part_isNeutralHadron'] = ak.values_astype((PID == 130) + (PID == 2112) + (PID == -2112), 'float32')
    v['part_isPhoton'] = ak.values_astype(PID == 22, 'float32')
    v['part_isEPlus'] = ak.values_astype(PID == -11, 'float32')
    v['part_isEMinus'] = ak.values_astype(PID == 11, 'float32')
    v['part_isMuPlus'] = ak.values_astype(PID == -13, 'float32')
    v['part_isMuMinus'] = ak.values_astype(PID == 13, 'float32')

    v['part_isChargedHadron'] = v['part_isCHPlus'] + v['part_isCHMinus']
    v['part_isElectron'] = v['part_isEPlus'] + v['part_isEMinus']
    v['part_isMuon'] = v['part_isMuPlus'] + v['part_isMuMinus']

    v['part_charge'] = (v['part_isCHPlus'] + v['part_isEPlus'] + v['part_isMuPlus']
                        ) - (v['part_isCHMinus'] + v['part_isEMinus'] + v['part_isMuMinus'])

    for k in list(v.keys()):
        if k.endswith('Plus') or k.endswith('Minus'):
            del v[k]

    return v


def convert(sources, destdir, basename):
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    for idx, sourcefile in enumerate(sources):
        npfile = np.load(sourcefile)
        output = os.path.join(destdir, '%s_%d.parquet' % (basename, idx))
        print(sourcefile)
        print(str(npfile['X'].shape))
        print(output)
        if os.path.exists(output):
            os.remove(output)
        v = _transform(npfile['X'], npfile['y'])
        arr = ak.Array(v)
        ak.to_parquet(arr, output, compression='LZ4', compression_level=4)


def natural_sort(l):
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert qg benchmark datasets')
    parser.add_argument('-i', '--inputdir', required=True, help='Directory of input numpy files.')
    parser.add_argument('-o', '--outputdir', required=True, help='Output directory.')
    parser.add_argument('--train-test-split', type=float, default=0.9, help='Training / testing split fraction.')
    args = parser.parse_args()

    import glob
    sources = natural_sort(glob.glob(os.path.join(args.inputdir, 'QG_jets*.npz')))
    n_train = int(args.train_test_split * len(sources))
    train_sources = sources[:n_train]
    test_sources = sources[n_train:]

    convert(train_sources, destdir=args.outputdir, basename='train_file')
    convert(test_sources, destdir=args.outputdir, basename='test_file')
