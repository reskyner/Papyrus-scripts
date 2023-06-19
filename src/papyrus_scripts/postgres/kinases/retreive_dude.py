from rdkit.Chem import PandasTools
from rdkit import Chem 
import pandas as pd
import urllib.request

from .classifier_pipeline import getMolDescriptors

def get_data(target, label):
    actives = PandasTools.LoadSDF(f'{target}_{label}.sdf.gz')
    actives['SMILES'] = actives.apply(lambda x: Chem.MolToSmiles(x['ROMol'], isomericSmiles=False), axis=1)
    actives.drop(columns='ROMol', inplace=True)
    actives.drop_duplicates(subset='ID', inplace=True)

    return actives


targets = [
    'abl1',
    'akt1',
    'akt2',
    'braf',
    'cdk2',
    'csf1r',
    'egfr',
    'fak1',
    'fgfr1',
    'igf1r',
    'jak2',
    'kit',
    'kpcb',
    'lck',
    'mapk2',
    'met',
    'mk01',
    'mk10',
    'mk14',
    'mp2k1',
    'plk1',
    'rock1',
    'src',
    'tgfr1',
    'vgfr2',
    'wee1'
]

for target in targets:
    urllib.request.urlretrieve(f"http://dude.docking.org/targets/{target}/decoys_final.sdf.gz", f"data/dude/{target}_decoys.sdf.gz")

td = []
for target in targets:
    # actives = get_data(target, 'actives')
    # actives['active'] = 1

    inactives = get_data(target, 'decoys')
    inactives['active'] = 0

    # td.append(actives)
    td.append(inactives)

all = pd.concat(td)

all.drop_duplicates(subset='ID', inplace=True)
all.dropna(inplace=True)

desc = all_df.apply(lambda x: getMolDescriptors(x['SMILES'], x['ID']), axis=1)
all_desc = all.merge(pd.DataFrame(list(desc)), left_on='ID', right_on='mol_id')

all_desc.to_csv('dude_decoys.csv', index=False)