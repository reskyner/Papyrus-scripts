from .models import *
from .session import get_db_session
from sqlalchemy import func
import tqdm
import pandas as pd
from rdkit import Chem
from joblib import dump
import glob

from .classifier_pipeline import getMolDescriptors

def get_smiles(mol):
    return Chem.MolToSmiles(mol)

session = get_db_session()

family_prots = {}

families = ['AGC', 'CAMK', 'CK1', 'CMGC', 'Other', 'STE', 'TK', 'TKL']

quals = ['High', 'Medium', 'Low']

family_dfs = {}

cols = ['activity__target_id', 'activity__accession', 'activity__pchembl_value',
       'activity__protein_type', 'activity__aid',
       'activity__pchembl_value_mean', 'activity__relation',
       'activity__pchembl_value_stdev', 'activity__pchembl_value_SEM',
       'activity__activity_id', 'activity__doc_id',
       'activity__pchembl_value_n', 'activity__papyrus_activity_id',
       'activity__year', 'activity__pchembl_value_median',
       'activity__pchembl_value_mad', 'molecule__smiles',
       'molecule__connectivity', 'molecule__inchi_key',
       'molecule__id', 'SMILES',
       'molecule__inchi', 'type', 'quality', 'target']

activity_cutoff = 6.5


for family in families:
    o = []
    q = session.query(Classification.classification).filter(func.lower(Classification.classification).contains('kinase')).filter(Classification.classification.contains(family)).all()
    others = [f for f in families if f!=family]
    classes = [c[0] for c in q]

    for c in classes:
        if not [c for c in c.split() if c in others]:
         o.append(c)
            
    kinase_prots = session.query(Protein).filter(Protein.classification.any(Classification.classification.in_(o))).all()
    family_prots[family] = [k.__dict__ for k in kinase_prots]


for family in families:
    print(family)
    for i in tqdm.tqdm(range(0, len(family_prots[family]))):
        activities = session.query(Activity, Molecule, ActivityType, Quality).join(Molecule).join(ActivityType).join(Quality).filter(Activity.target_id==family_prots[family][i]['target_id']).all()
        dict_tups = [(a.__dict__, m.__dict__, at.__dict__, q.__dict__) for a,m,at,q in activities]
        d_list = []
        for tup in dict_tups:
            o = {}
            for key, item in tup[0].items():
                if not key.startswith('_') and key not in ['id', 'quality', 'type', 'molecule_id']:
                    o[f'activity__{key}'] = item
            for key, item in tup[1].items():
                if not key.startswith('_'):
                    o[f'molecule__{key}'] = item
            for key, item in tup[2].items():
                if not key.startswith('_') and key!='id':
                    o[f'{key}'] = item
            for key, item in tup[3].items():
                if not key.startswith('_') and key!='id':
                    o[f'{key}'] = item
            d_list.append(o)
        family_prots[family][i]['molecules'] = d_list


for family, proteins in family_prots.items():
    print(family)
    for i in tqdm.tqdm(range(0, len(proteins))):
        molecule_list = proteins[i]['molecules']
        for molecule in molecule_list:
            molecule['target'] = proteins[i]['target_id']
    all_mols = [proteins[i]['molecules'] for i in range(0, len(proteins))]
    flat_list = [item for sublist in all_mols for item in sublist]
    family_df = pd.DataFrame(flat_list)
    print('generating smiles')
    family_df['SMILES'] = family_df.apply(lambda x: get_smiles(x['molecule__mol']), axis=1)
    print('writing csv')
    family_df.to_csv(f'{family}_unfiltered.csv', columns=cols, index=False)
    print('copying df')
    family_dfs[family] = family_df
    print('\n')

del family_prots

dump(family_dfs, 'family_dfs.joblib')

for q in quals:
    for family in families:
        
        actives = family_dfs[family][family_dfs[family]['type']!='other'][family_dfs[family]['activity__relation']!='<'][family_dfs[family]['activity__pchembl_value']>activity_cutoff][family_dfs[family]['quality']==q]
        inactives_a = family_dfs[family][family_dfs[family]['type']!='other'][family_dfs[family]['activity__relation']=='<'][family_dfs[family]['quality']==q]
        inactives_b = family_dfs[family][family_dfs[family]['type']!='other'][family_dfs[family]['activity__relation']!='<'][family_dfs[family]['activity__pchembl_value']<activity_cutoff][family_dfs[family]['quality']==q]
        inactives = pd.concat([inactives_a, inactives_b])
    
        all = list(set(pd.concat([actives, inactives])['molecule__id']))
        both = [i for i in all if i in inactives['molecule__id'] and i in actives['molecule__id']]
        actives = actives[~actives['molecule__id'].isin(both)]
        inactives = inactives[~inactives['molecule__id'].isin(both)]

        mol_ids = list(set(actives['molecule__id']))
        counts = actives['molecule__id'].value_counts()
        keep = []
        for m in mol_ids:
            if counts[m]>1 and len(list(set(actives[actives['molecule__id']==m]['activity__target_id'])))>1:
                keep.append(m)
        actives = actives[actives['molecule__id'].isin(keep)]
        actives.to_csv(f'{family}_actives_{q}.csv', index=False)

        mol_ids = list(set(inactives['molecule__id']))
        counts = inactives['molecule__id'].value_counts()
        keep = []
        for m in mol_ids:
            if counts[m]>1 and len(list(set(inactives[inactives['molecule__id']==m]['activity__target_id'])))>1:
                keep.append(m)
        
        inactives = inactives[inactives['molecule__id'].isin(keep)]
        inactives.to_csv(f'{family}_inactives_{q}.csv', index=False, columns=cols)

dfs = {}

actives = glob.glob('*actives*.csv')
print(actives)
for fname in actives:
    family, cat, qual = fname.split('_')
    qual = qual.split('.')[0]
    if family not in dfs.keys():
        dfs[family] = {}
    if cat not in dfs[family].keys():
        dfs[family][cat] = {}
    # if qual not in dfs[family][cat].keys():
    dfs[family][cat][qual] = pd.read_csv(fname)


for family in dfs.keys():
    for quality in ['Low', 'Medium', 'High']:
        print(family)
        actives_high = pd.DataFrame(list(set(list(zip(dfs[family]['actives'][quality]['molecule__id'], dfs[family]['actives'][quality]['SMILES'])))), columns=['mol_id', 'SMILES'])
        actives_high['active'] = 1
        inactives_high = pd.DataFrame(list(set(list(zip(dfs[family]['inactives'][quality]['molecule__id'], dfs[family]['inactives'][quality]['SMILES'])))), columns=['mol_id', 'SMILES'])
        inactives_high['active'] = 0
    
        all_high = pd.concat([actives_high, inactives_high])
    
        print('calculating descriptors')
        desc_dicts = all_high.apply(lambda x: getMolDescriptors(x['SMILES'], x['mol_id']), axis=1)
        all_high = all_high.merge(pd.DataFrame(list(desc_dicts)), left_on='mol_id', right_on='mol_id')
    
        all_high.to_csv(f'descriptors_{family}_{quality}.csv')




