import os
import random
import torch
import numpy as np
from os.path import join, exists
from torch import nn
import json
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from numba import jit

SEED = 123

TARGET_LABELS = [
    '5-alpha_reductase_inhibitor', '11-beta-hsd1_inhibitor', 'acat_inhibitor', 'acetylcholine_receptor_agonist',
    'acetylcholine_receptor_antagonist', 'acetylcholinesterase_inhibitor', 'adenosine_receptor_agonist',
    'adenosine_receptor_antagonist', 'adenylyl_cyclase_activator', 'adrenergic_receptor_agonist',
    'adrenergic_receptor_antagonist', 'akt_inhibitor', 'aldehyde_dehydrogenase_inhibitor', 'alk_inhibitor',
    'ampk_activator', 'analgesic', 'androgen_receptor_agonist', 'androgen_receptor_antagonist', 'anesthetic_-_local',
    'angiogenesis_inhibitor', 'angiotensin_receptor_antagonist', 'anti-inflammatory', 'antiarrhythmic', 'antibiotic',
    'anticonvulsant', 'antifungal', 'antihistamine', 'antimalarial', 'antioxidant', 'antiprotozoal', 'antiviral',
    'apoptosis_stimulant', 'aromatase_inhibitor', 'atm_kinase_inhibitor', 'atp-sensitive_potassium_channel_antagonist',
    'atp_synthase_inhibitor', 'atpase_inhibitor', 'atr_kinase_inhibitor', 'aurora_kinase_inhibitor',
    'autotaxin_inhibitor', 'bacterial_30s_ribosomal_subunit_inhibitor', 'bacterial_50s_ribosomal_subunit_inhibitor',
    'bacterial_antifolate', 'bacterial_cell_wall_synthesis_inhibitor', 'bacterial_dna_gyrase_inhibitor',
    'bacterial_dna_inhibitor', 'bacterial_membrane_integrity_inhibitor', 'bcl_inhibitor', 'bcr-abl_inhibitor',
    'benzodiazepine_receptor_agonist', 'beta_amyloid_inhibitor', 'bromodomain_inhibitor', 'btk_inhibitor',
    'calcineurin_inhibitor', 'calcium_channel_blocker', 'cannabinoid_receptor_agonist',
    'cannabinoid_receptor_antagonist', 'carbonic_anhydrase_inhibitor', 'casein_kinase_inhibitor', 'caspase_activator',
    'catechol_o_methyltransferase_inhibitor', 'cc_chemokine_receptor_antagonist', 'cck_receptor_antagonist',
    'cdk_inhibitor', 'chelating_agent', 'chk_inhibitor', 'chloride_channel_blocker', 'cholesterol_inhibitor',
    'cholinergic_receptor_antagonist', 'coagulation_factor_inhibitor', 'corticosteroid_agonist',
    'cyclooxygenase_inhibitor', 'cytochrome_p450_inhibitor', 'dihydrofolate_reductase_inhibitor',
    'dipeptidyl_peptidase_inhibitor', 'diuretic', 'dna_alkylating_agent', 'dna_inhibitor', 'dopamine_receptor_agonist',
    'dopamine_receptor_antagonist', 'egfr_inhibitor', 'elastase_inhibitor', 'erbb2_inhibitor',
    'estrogen_receptor_agonist', 'estrogen_receptor_antagonist', 'faah_inhibitor', 'farnesyltransferase_inhibitor',
    'fatty_acid_receptor_agonist', 'fgfr_inhibitor', 'flt3_inhibitor', 'focal_adhesion_kinase_inhibitor',
    'free_radical_scavenger', 'fungal_squalene_epoxidase_inhibitor', 'gaba_receptor_agonist',
    'gaba_receptor_antagonist', 'gamma_secretase_inhibitor', 'glucocorticoid_receptor_agonist', 'glutamate_inhibitor',
    'glutamate_receptor_agonist', 'glutamate_receptor_antagonist', 'gonadotropin_receptor_agonist', 'gsk_inhibitor',
    'hcv_inhibitor', 'hdac_inhibitor', 'histamine_receptor_agonist', 'histamine_receptor_antagonist',
    'histone_lysine_demethylase_inhibitor', 'histone_lysine_methyltransferase_inhibitor', 'hiv_inhibitor',
    'hmgcr_inhibitor', 'hsp_inhibitor', 'igf-1_inhibitor', 'ikk_inhibitor', 'imidazoline_receptor_agonist',
    'immunosuppressant', 'insulin_secretagogue', 'insulin_sensitizer', 'integrin_inhibitor', 'jak_inhibitor',
    'kit_inhibitor', 'laxative', 'leukotriene_inhibitor', 'leukotriene_receptor_antagonist', 'lipase_inhibitor',
    'lipoxygenase_inhibitor', 'lxr_agonist', 'mdm_inhibitor', 'mek_inhibitor', 'membrane_integrity_inhibitor',
    'mineralocorticoid_receptor_antagonist', 'monoacylglycerol_lipase_inhibitor', 'monoamine_oxidase_inhibitor',
    'monopolar_spindle_1_kinase_inhibitor', 'mtor_inhibitor', 'mucolytic_agent', 'neuropeptide_receptor_antagonist',
    'nfkb_inhibitor', 'nicotinic_receptor_agonist', 'nitric_oxide_donor', 'nitric_oxide_production_inhibitor',
    'nitric_oxide_synthase_inhibitor', 'norepinephrine_reuptake_inhibitor', 'nrf2_activator', 'opioid_receptor_agonist',
    'opioid_receptor_antagonist', 'orexin_receptor_antagonist', 'p38_mapk_inhibitor', 'p-glycoprotein_inhibitor',
    'parp_inhibitor', 'pdgfr_inhibitor', 'pdk_inhibitor', 'phosphodiesterase_inhibitor', 'phospholipase_inhibitor',
    'pi3k_inhibitor', 'pkc_inhibitor', 'potassium_channel_activator', 'potassium_channel_antagonist',
    'ppar_receptor_agonist', 'ppar_receptor_antagonist', 'progesterone_receptor_agonist',
    'progesterone_receptor_antagonist', 'prostaglandin_inhibitor', 'prostanoid_receptor_antagonist',
    'proteasome_inhibitor', 'protein_kinase_inhibitor', 'protein_phosphatase_inhibitor', 'protein_synthesis_inhibitor',
    'protein_tyrosine_kinase_inhibitor', 'radiopaque_medium', 'raf_inhibitor', 'ras_gtpase_inhibitor',
    'retinoid_receptor_agonist', 'retinoid_receptor_antagonist', 'rho_associated_kinase_inhibitor',
    'ribonucleoside_reductase_inhibitor', 'rna_polymerase_inhibitor', 'serotonin_receptor_agonist',
    'serotonin_receptor_antagonist', 'serotonin_reuptake_inhibitor', 'sigma_receptor_agonist',
    'sigma_receptor_antagonist', 'smoothened_receptor_antagonist', 'sodium_channel_inhibitor',
    'sphingosine_receptor_agonist', 'src_inhibitor', 'steroid', 'syk_inhibitor', 'tachykinin_antagonist',
    'tgf-beta_receptor_inhibitor', 'thrombin_inhibitor', 'thymidylate_synthase_inhibitor', 'tlr_agonist',
    'tlr_antagonist', 'tnf_inhibitor', 'topoisomerase_inhibitor', 'transient_receptor_potential_channel_antagonist',
    'tropomyosin_receptor_kinase_inhibitor', 'trpv_agonist', 'trpv_antagonist', 'tubulin_inhibitor',
    'tyrosine_kinase_inhibitor', 'ubiquitin_specific_protease_inhibitor', 'vegfr_inhibitor', 'vitamin_b',
    'vitamin_d_receptor_agonist', 'wnt_inhibitor'
]

class Settings:
    def __init__(self, filepath):
        self.filepath = filepath
        self.settings = json.load(open(self.filepath))

    def __getitem__(self, idx):
        return self.settings[idx]


class Config:
    def __init__(self, filepath):
        self.filepath = filepath
        self.module_path = self.filepath.split('.')[0].replace('/', '.')
        self.config_name = self.module_path.split('.')[-1]
        self.config = getattr(__import__(self.module_path), self.config_name).config

    def __getitem__(self, idx):
        return self.config[idx]


class Metrics:
    def __init__(self):
        self.data = []

    def load(self, data):
        self.data = data

    def add(self, x):
        return self.data.append(x)

    def mean(self):
        return np.mean(self.data)

    def min_epoch(self):
        return np.argmin(self.data)

    def min(self):
        return np.min(self.data)

    def max(self):
        return np.max(self.data)

    def tail(self):
        return self.data[-1]

    def reset(self):
        self.data = []



def seed_everything():
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split_data(x, y, n_splits):
    kfold = MultilabelStratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
    for train_index, valid_index in kfold.split(x, y):
        yield (x.iloc[train_index], y.iloc[train_index]), (x.iloc[valid_index], y.iloc[valid_index])


def log_loss(prediction, target, cuda=True):
    prediction = torch.Tensor(prediction)
    target = torch.Tensor(target)
    if cuda:
        prediction = prediction.cuda().float()
        target = target.cuda().float()
    criterion = nn.BCELoss()
    loss = criterion(prediction, target)
    return loss.detach().cpu().numpy()


def print_table(data, row_prefix, col_prefix, title):
    data = np.array(data)
    data = np.vstack([data, np.mean(data, axis=0)])
    data = np.hstack([data, np.expand_dims(np.mean(data, axis=1), axis=1)])

    rows = []
    for row_i, row in enumerate(data):
        if row_i == len(data) - 1:
            prefix = 'Average'
        else:
            prefix = '%s_%d' % (row_prefix, row_i)
        row_text = '%s %s' % (prefix, ''.join([' %.5f' % col for col in row]))
        rows.append(row_text)
    text = '\n'.join(rows)

    spacer = '-' * int((len(text.split('\n')[0]) - 2 - len(title)) / 2)
    margin = ' ' * int(len(row_prefix) + 1)
    title_text = '%s %s %s\n %s' % (spacer, title, spacer, margin)
    for col_i in range(len(data[0])):
        if col_i == len(data[0]) - 1:
            prefix = 'Average'
        else:
            prefix = '%s_%d' % (col_prefix, col_i)
        title_text += '  %s' % prefix

    print(title_text + '\n' + text)


SETTINGS = Settings('SETTINGS.json')
