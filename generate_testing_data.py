from pycbc.waveform.waveform import get_fd_waveform
import matplotlib.pyplot as plt
import os
from pathlib import Path
from multiprocessing import Pool
import sys
import gwmat
import bilby
import pandas as pd
import numpy as np
from gwpy.timeseries import TimeSeries as ts

sys.stdout = open("log.out", "w")
sys.stderr = open("error.err", "w")

num_processes = int(sys.argv[1])
num = int(sys.argv[2])

# Accessing directories

os.makedirs('testing', exist_ok=True)
testing_data_path = Path("testing/")

os.makedirs(testing_data_path / 'lensed', exist_ok=True)
os.makedirs(testing_data_path / 'unlensed', exist_ok=True)
lensed_path = testing_data_path / "lensed"
unlensed_path = testing_data_path / "unlensed"

# Creating samples for testing data

prior = bilby.gw.prior.BBHPriorDict()
prior['Log_Mlz'] = bilby.core.prior.Uniform(minimum = 3, maximum = 5)
prior['yl'] = bilby.core.prior.PowerLaw(alpha = 1, minimum = 0.01, maximum = 1)
prior['phase'] = bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")
prior['geocent_time'] = bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)

priors = prior.sample(num)

df = list((pd.DataFrame(priors)).iterrows())

param_result = [result[0] for result in map(lambda x: bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(x[1]), df)]

# Generating testing data
def generate_testing_qtransform(num):
    init_params = dict(wf_domain="TD", f_start=15, snr_f_min=20., snr_f_max=None,
                f_ref=20., sample_rate=2048, delta_t=None, wf_approximant="IMRPhenomPv2",
                ifo_list = ['H1', 'L1', 'V1'])
    param_gen = param_result[num].copy()
    param_gen.pop('mass_ratio')
    param_gen.pop('chirp_mass')
    param_gen.pop('total_mass')
    m_lens=np.power(10., param_gen.pop("Log_Mlz"))
    y_lens=param_gen.pop("yl")
    lens_params = dict(m_lens=m_lens, y_lens=y_lens, z_lens=0.,
                    lens_mass_lower_limit=1.e-3, Ff_data=None)
    cbc_params =  param_gen
    misc_params = dict(rwrap = -0.1, cyclic_time_shift_method = "gwmat",
                    taper_hp_hc=True,  hp_hc_extra_padding_at_start=0,
                    make_hp_hc_duration_power_of_2=True,
                    extra_padding_at_start=1, extra_padding_at_end=2,
                    save_data=False, data_outdir = './',
                    data_label=None, data_channel='PyCBC-Injection')
    psd_params = dict(Noise=True, psd_H1="O4", psd_L1="O4", psd_V1="O4", 
                    noise_seed=None, is_asd_file=False, psd_f_low=None)

    lensing_params = {**init_params, **cbc_params, **lens_params, **psd_params, **misc_params}

    nolens_params = dict(m_lens=0, y_lens=0, z_lens=0.,
                    lens_mass_lower_limit=1.e-3, Ff_data=None)
    
    unlensing_params = {**init_params, **cbc_params, **nolens_params, **psd_params, **misc_params}

    lens_res = gwmat.injection.simulate_injection_with_comprehensive_output(**lensing_params)
    unlens_res = gwmat.injection.simulate_injection_with_comprehensive_output(**unlensing_params)

    wf_lens_h = lens_res['noisy_ifo_signal']['H1']
    wf_unlens_h = unlens_res['noisy_ifo_signal']['H1']

    wf_lens_l = lens_res['noisy_ifo_signal']['L1']
    wf_unlens_l = unlens_res['noisy_ifo_signal']['L1']

    wf_lens_v = lens_res['noisy_ifo_signal']['V1']
    wf_unlens_v = unlens_res['noisy_ifo_signal']['V1']

    if lens_res['matched_filter_snr']['H1'] > 10.0:
        wf_lens_h = ts.from_pycbc(wf_lens_h)
        wf_unlens_h = ts.from_pycbc(wf_unlens_h)

        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(wf_lens_h.q_transform(logf=True, norm='mean', outseg=(-1.2, 0.2), frange=(20,512), whiten=False, qrange=(4, 64)))  
        plt.axis("off")
        plt.savefig(lensed_path / f"$M_l$_{m_lens}_y_{y_lens}_h.png")
        plt.close()

        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(wf_unlens_h.q_transform(logf=True, norm='mean', outseg=(-1.2, 0.2), frange=(20,512), whiten=False, qrange=(4, 64)))  
        plt.axis("off")
        plt.savefig(unlensed_path / f"$M_l$_{m_lens}_y_{y_lens}_h.png")
        plt.close()

    if lens_res['matched_filter_snr']['L1'] > 10.0:
        wf_lens_l = ts.from_pycbc(wf_lens_l)
        wf_unlens_l = ts.from_pycbc(wf_unlens_l)

        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(wf_lens_l.q_transform(logf=True, norm='mean', outseg=(-1.2, 0.2), frange=(20,512), whiten=False, qrange=(4, 64)))  
        plt.axis("off")
        plt.savefig(lensed_path / f"$M_l$_{m_lens}_y_{y_lens}_l.png")
        plt.close()

        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(wf_unlens_l.q_transform(logf=True, norm='mean', outseg=(-1.2, 0.2), frange=(20,512), whiten=False, qrange=(4, 64)))  
        plt.axis("off")
        plt.savefig(unlensed_path / f"$M_l$_{m_lens}_y_{y_lens}_l.png")
        plt.close()
        
    if lens_res['matched_filter_snr']['V1'] > 10.0:
        wf_lens_v = ts.from_pycbc(wf_lens_v)
        wf_unlens_v = ts.from_pycbc(wf_unlens_v)

        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(wf_lens_v.q_transform(logf=True, norm='mean', outseg=(-1.2, 0.2), frange=(20,512), whiten=False, qrange=(4, 64)))  
        plt.axis("off")
        plt.savefig(lensed_path / f"$M_l$_{m_lens}_y_{y_lens}_v.png")
        plt.close()

        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(wf_unlens_v.q_transform(logf=True, norm='mean', outseg=(-1.2, 0.2), frange=(20,512), whiten=False, qrange=(4, 64)))  
        plt.axis("off")
        plt.savefig(unlensed_path / f"$M_l$_{m_lens}_y_{y_lens}_v.png")
        plt.close()

num_range = list(range(int(num)))

with Pool(processes=num_processes) as pool:
        qtransforms = pool.map(generate_testing_qtransform, num_range)

sys.stdout.close()
sys.stderr.close()

