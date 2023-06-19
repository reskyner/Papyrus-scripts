import requests
import pandas as pd
from rdkit import Chem
from .classifier_pipeline import getMolDescriptors

response = requests.get('https://klifs.net/api/kinase_groups')
kinase_groups = response.json()

kinase_names = {}

for grp in kinase_groups:
    grp_url = f'https://klifs.net/api/kinase_names?kinase_group={grp}'
    response = requests.get(grp_url)
    result = response.json()

    kinase_names[grp] = result

failed_ids = []

for key in kinase_names.keys():
    for i in kinase_names[key]:
        kid = i['kinase_ID']
        url = f'https://klifs.net/api/ligands_list?kinase_ID={kid}'
        response = requests.get(url)
        res = response.json()
        if res[0]==400:
            failed_ids.append(kid)
            i['ligs'] = []
        else:
            i['ligs'] = res

pdb_classes = {}

for key in kinase_names.keys():
    for kinase in kinase_names[key]:
        # print(kinase)
        for lig in kinase['ligs']:
            if lig['PDB-code'] in pdb_classes.keys():
                if not key in pdb_classes[lig['PDB-code']]:
                    pdb_classes[lig['PDB-code']].append(key)
            else:
                pdb_classes[lig['PDB-code']] = [key]

all_ids = pdb_classes.keys()
df = pd.DataFrame()
df['lig_id'] = list(all_ids)

id_smi = {}

for key in kinase_names.keys():
    for kinase in kinase_names[key]:
        # print(kinase)
        for lig in kinase['ligs']:
           if lig['PDB-code'] not in id_smi:
               m = Chem.MolFromSmiles(lig['SMILES'])
               if m:
                   id_smi[lig['PDB-code']] = lig['SMILES']
               else:
                   continue
               
drop = []

for i in df['lig_id']:
    if i not in id_smi.keys():
        drop.append(i)

df = df[~df['lig_id'].isin(drop)]
df.reset_index(inplace=True, drop=True)

def get_smi(row):
    return id_smi[row['lig_id']]

df['SMILES'] = df.apply(lambda x: get_smi(x), axis=1)
desc_dicts = df.apply(lambda x: getMolDescriptors(x['SMILES'], x['lig_id']), axis=1)

desc_df = df.merge(pd.DataFrame(list(desc_dicts)), left_on='lig_id', right_on='lig_id')
desc_df.to_csv('klifs_class_data-ligs.csv', index=False)