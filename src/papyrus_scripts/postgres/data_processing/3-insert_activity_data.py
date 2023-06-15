import sys
sys.path.append('/code')

from papyrus.postgres.database.models import (Protein, Organism, Classification, Molecule, Activity, ActivityType, Source, Quality, CID)

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
import os
import glob

import pandas as pd
import numpy as np
import json

import multiprocessing
import gc
from tqdm import tqdm
from copy import copy
import numpy as np

from rdkit import Chem

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
    

# get neccessary molecule data as list of tuples
session = get_db_session()
mol_data = session.query(Molecule.id, Molecule.smiles, Molecule.inchi_auxinfo).all()
mol_ids = {}
for t in tqdm(mol_data):
    combo = f'{t[1]}_{t[2]}'
    if not combo in mol_ids.keys():
        mol_ids[combo] = [t[0]]
    else:
        mol_ids[combo].append(t[0])

# add activity types
types = ['EC50', 'IC50', 'KD', 'Ki', 'other']
atypes = [ActivityType(type=t) for t in types]
session.add_all(atypes)
session.commit()

# return ids for activity types as dict
types = dict(session.query(ActivityType.type, ActivityType.id).all())

# add qualities
qualities = ["High","Low","Medium","Medium;Low","Low;Medium"]
qs = [Quality(quality=q) for q in qualities]
session.add_all(qs)
session.commit()

# return ids for qualities as list of tuples
qualities = dict(session.query(Quality.quality,Quality.id).all())

# get all targets - probably don't need this 
targets = [t[0] for t in session.query(Protein.target_id).all()]
# print(list(targets))

pchembl_val_list = lambda pchembl_values: ([v.rstrip() for v in pchembl_values.split(';')])
smiles = lambda smi: (Chem.CanonSmiles(smi))

converters = {
    'SMILES':smiles,
    'pchembl_value':pchembl_val_list
}

dtype_file = '../.data/data_types.json'
activity_data = '../.data/05.6_combined_set_without_stereochemistry.tsv.xz'

with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)['papyrus']

# all columns needed to process activity data
activity_columns = ['Activity_ID', 'SMILES', 'InChI_AuxInfo', 'accession', 'Protein_Type', 'AID', 'doc_id',
                    'Year', 'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'relation', 'pchembl_value',
                    'pchembl_value_Mean', 'pchembl_value_StdDev', 'pchembl_value_SEM', 'pchembl_value_N', 
                    'pchembl_value_Median', 'pchembl_value_MAD','Quality'
                   ]

activity_reader = pd.read_csv(activity_data, 
                     sep='\t', 
                     compression='xz', 
                     chunksize = 10000, 
                     iterator=True, 
                     dtype=dtypes,
                     converters=converters,
                     usecols=activity_columns
                    )


from datetime import datetime

def sanitize_and_split(row, length, spl=';'):
    split = [v.rstrip() for v in str(row).split(spl)]
    if len(split)!= length:
        split = [split[0] for i in range(0,length)]
    
    split = [None if x == '' else x for x in split]
    
    return split

def get_activity_dicts(row):
    slice_list = []
    pchembl_values = [v.rstrip() for v in row.pchembl_value]

    aids = sanitize_and_split(row=row.AID,length=row.pchembl_len)        
    doc_ids = sanitize_and_split(row=row.all_doc_ids,length=row.pchembl_len)
    years = sanitize_and_split(row=row.all_years,length=row.pchembl_len)
    type_IC50s = sanitize_and_split(row=row.type_IC50,length=row.pchembl_len)         
    type_EC50s = sanitize_and_split(row=row.type_EC50,length=row.pchembl_len)
    type_KDs = sanitize_and_split(row=row.type_KD,length=row.pchembl_len)
    type_Kis = sanitize_and_split(row=row.type_Ki,length=row.pchembl_len)

    for j in range(0, row.pchembl_len):
        update_dict = {
            'pchembl_value': pchembl_values[j],
            'AID': aids[j],
            'doc_id': doc_ids[j],
            'Year': years[j],
            'type_IC50': type_IC50s[j],
            'type_EC50': type_EC50s[j],
            'type_KD': type_KDs[j],
            'type_Ki': type_Kis[j]
        }
        row_copy = copy(row._asdict())

        row_copy.update(update_dict)

        slice_list.append(row_copy)
            
    return slice_list

activity_type_map = {
        '1000':'IC50',
        '0100':'EC50',
        '0010':'KD',
        '0001':'Ki',
        '0000':'other',
    }

def get_atype(binstr):
    return activity_type_map[binstr]

activity_columns = ['Activity_ID', 'SMILES', 'InChI_AuxInfo', 'accession', 'Protein_Type', 'AID', 'doc_id',
                    'Year', 'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'relation', 'pchembl_value',
                    'pchembl_value_Mean', 'pchembl_value_StdDev', 'pchembl_value_SEM', 'pchembl_value_N', 
                    'pchembl_value_Median', 'pchembl_value_MAD','Quality', 'all_doc_ids', 'all_years',
                    'target_id'
                   ]

activity_reader = pd.read_csv(activity_data, 
                     sep='\t', 
                     compression='xz', 
                     chunksize = 50000, 
                     iterator=True, 
                     dtype=dtypes,
                     converters=converters,
                     usecols=activity_columns,
                     # skiprows=[i for i in range(1,2950000)]
                    )

xlen = lambda x: (len(x))
sval = lambda x: (x[0])


def get_molid(key):
    return mol_ids[key][0]

mid = lambda x: get_molid(f'{x["SMILES"]}_{x["InChI_AuxInfo"]}')

import math

def yval(val):
    if not val:
        return 0
    if math.isnan(float(val)):
        return 0
    else:
        return int(val)
    
missing_targets = []
    

yvalx = lambda x: (yval(x))
print(targets)
for i,df in enumerate(activity_reader):
    if i<756:
        continue
    print(i*50000)
    now = datetime.now()        
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    df['pchembl_len'] = df['pchembl_value'].map(xlen)
    multis = df[df['pchembl_len']>1]
    updated_rows = []
    for row in multis.itertuples():
        updated_rows.extend(get_activity_dicts(row))

    multi_expanded = pd.DataFrame(updated_rows)
    singles = df[df['pchembl_len']==1]
    singles['pchembl_value'] = singles['pchembl_value'].map(sval)
    all_expanded = pd.concat([multi_expanded, singles])
    
    all_expanded['atype'] = all_expanded.apply(lambda x: get_atype(f"{x['type_IC50']}{x['type_EC50']}{x['type_KD']}{x['type_Ki']}"), axis=1)
    all_expanded['molecule_id'] = all_expanded.apply(mid, axis=1)
    all_expanded['quality'] = all_expanded.apply(lambda x: (qualities[x['Quality']]), axis=1)
    all_expanded['activity_id'] = all_expanded.apply(lambda x: (types[x['atype']]), axis=1)
    all_expanded['year'] = all_expanded['Year'].map(yvalx)
    
    all_expanded.rename(columns = {
                                    'Activity_ID': 'papyrus_activity_id',
                                    'Protein_Type': 'protein_type',
                                    'AID': 'aid',
                                    'activity_id': 'type',
                                    'pchembl_value_Mean': 'pchembl_value_mean',
                                    'pchembl_value_StdDev': 'pchembl_value_stdev',
                                    'pchembl_value_N': 'pchembl_value_n',
                                    'pchembl_value_Median': 'pchembl_value_median',
                                    'pchembl_value_MAD': 'pchembl_value_mad'
                                   }, inplace = True)
    
    
    to_keep = ['papyrus_activity_id', 
       'target_id', 'accession', 'protein_type', 'aid', 'doc_id', 
        'relation', 'pchembl_value', 'pchembl_value_mean',
       'pchembl_value_stdev', 'pchembl_value_SEM', 'pchembl_value_n',
       'pchembl_value_median', 'pchembl_value_mad',
       'molecule_id', 'quality', 'type', 'year']
    
    to_del = [x for x in all_expanded.columns if x not in to_keep]
    
    all_expanded.drop(to_del, axis=1, inplace=True)
    all_expanded.fillna('', inplace=True)
    all_expanded.replace(r'^\s*$', None, regex=True, inplace=True)
    
    adicts = all_expanded.to_dict('records')
    clean_adicts = [{k:v if not pd.isnull(v) else None for k,v in d.items()} for d in adicts]
    
    to_add = []
    
    for d in clean_adicts:
        if d['target_id'] not in targets:
            missing_targets.append(d['target_id'])
        else:
            to_add.append(d)
    activities = [Activity(**d) for d in to_add]
    session.add_all(activities)
    session.commit()
    now = datetime.now()        
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

with open('missing_targets.txt', 'w') as w:
    for t in list(set(missing_targets)):
        w.write(f'{t}\n')
