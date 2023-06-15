import sys
sys.path.append('/code')
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
import os
from papyrus.postgres.database.models import Molecule

import pandas as pd
from rdkit import Chem
from razi.rdkit_postgresql.functions import morganbv_fp
import json
import gc
from tqdm import tqdm


from datetime import datetime

def get_db_session():
    engine = create_engine(
        'postgresql://postgres:postgres@localhost:5432/papyrus',
        pool_recycle=3600, pool_size=10)
    db_session = scoped_session(sessionmaker(
        autocommit=False, autoflush=False, bind=engine))
    
    return db_session

class TypeDecoder(json.JSONDecoder):
    """Custom json decoder to support types as values."""

    def __init__(self, *args, **kwargs):
        """Simple json decoder handling types as values."""
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Handle types."""
        if '__type__' not in obj:
            return obj
        module = obj['__type__']['module']
        type_ = obj['__type__']['type']
        if module == 'builtins':
            return getattr(__builtins__, type_)
        loaded_module = importlib.import_module(module)
        return getattr(loaded_module, type_)

dtype_file = '../.data/data_types.json'
activity_data = '../.data/05.6_combined_set_without_stereochemistry.tsv.xz'

with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)['papyrus']


smi_mol_fp = lambda smiles: (Chem.CanonSmiles(smiles))
cid_arr = lambda cids: (cids.split(';'))
source_arr = lambda sources: (sources.split(';'))

mol_cols = ['SMILES', 'InChI', 'CID', 'InChIKey', 'connectivity', 'InChI_AuxInfo', 'source']
converters = {
    'SMILES': smi_mol_fp,
    'CID': cid_arr,
    'source': source_arr
}

molecule_reader = pd.read_csv(activity_data, 
                     sep='\t', 
                     compression='xz', 
                     chunksize = 100000, 
                     iterator=True, 
                     dtype=dtypes,
                     converters=converters,
                     usecols=mol_cols
                    )

# mol_dfs = []

committed = []
smiles_cids = {}


for i,df in enumerate(molecule_reader):
    
    print(i)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
#     mol_dfs.append(df)
    
    idx_dict = {}

    print('getting unique combos')
    for i, row in tqdm(df.iterrows()):
        smi = row['SMILES']
        aux_info = row['InChI_AuxInfo']
        combo = f'{smi}_{aux_info}'
        if not combo in idx_dict.keys():
            idx_dict[combo] = [i]
        else:
            idx_dict[combo].append(i)

    mol_dicts = []

    print('creating mol dicts')
    for key in tqdm(idx_dict.keys()):
        if key in committed:
            continue
        rows = [df.loc[i] for i in idx_dict[key]]
        unique_auxinfos = list(set([series['InChI_AuxInfo'] for series in rows]))
        if len(unique_auxinfos)==1:
            all_cid_sources = []
            cid_sources = [list(zip(series['CID'], series['source'])) for series in rows]
            [all_cid_sources.extend(x) for x in cid_sources]
            unique_cid_sources = list(set(all_cid_sources))
            smiles_cids[key] = unique_cid_sources
            r = rows[0]
            md = {
                'smiles':r['SMILES'],
                'mol':'',
                'fp':'',
                'connectivity':r['connectivity'],
                'inchi_key':r['InChIKey'],
                'inchi':r['InChI'],
                'inchi_auxinfo':r['InChI_AuxInfo']
            }
            mol_dicts.append(md)
        else:
            print('something wrong')
            
    fdf = pd.DataFrame(mol_dicts)
    molobj = lambda smiles: (Chem.MolFromSmiles(smiles))
    fpobj = lambda smiles: (morganbv_fp(smiles))

    fdf['fp'] = fdf['smiles'].map(fpobj)
    fdf['mol'] = fdf['smiles'].map(molobj)

    table_dicts = fdf.to_dict('records')
    mols = [Molecule(**d) for d in table_dicts]

    sesh = get_db_session()
    sesh.add_all(mols)
    sesh.commit()
    sesh.close()
    sesh.remove()
    
    committed.extend(idx_dict.keys())

    now = datetime.now()        
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)
