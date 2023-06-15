import sys
sys.path.append('/code')

import pandas as pd
import json
import math
from database.models import (Protein, Organism, Classification, Molecule, Activity, ActivityType, Source, Quality, CID)

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

dtype_file = '../.data/data_types.json'
protein_data = '../.data/05.6_combined_set_protein_targets.tsv.xz'

def get_db_session():
    engine = create_engine(
        'postgresql://postgres:postgres@10.8.1.25:5432/papyrus',
        pool_recycle=3600, pool_size=10)
    db_session = scoped_session(sessionmaker(
        autocommit=False, autoflush=False, bind=engine))
    
    return db_session


def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        session.flush()
        session.refresh(instance)
        return instance
    
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


with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)['papyrus']
        
protein_df = pd.read_csv(protein_data, sep='\t', dtype=dtypes)

db_session = get_db_session()

rows = []
tids = []


for i, row in protein_df.iterrows():

    if row['target_id'] in tids:
        continue
    try:
        x = float(row['Organism'])
        if math.isnan(x):
            continue
    except:
        pass
    if not row['Organism'] or row['Organism']=='nan':
         continue
    organism = get_or_create(session=db_session, model=Organism, organism=row['Organism'])
    if not organism.id:
        print(row)
        continue
    classifications_list = str(row['Classification']).split('->')
    classifications = [get_or_create(session=db_session, model=Classification, classification=c) for c in classifications_list]
    
    review_mapping = {'reviewed':1, 'Unreviewed':0, 'unreviewed':0}
    
    prot = Protein(
        target_id = row['target_id'],
        HGNC_symbol = str(row['HGNC_symbol']),
        uniprot_id = row['UniProtID'],
        reviewed = review_mapping[row['Status']],
        organism = organism.id,
        length = row['Length'],
        sequence = row['Sequence'], 
        classifications = classifications
    )

       
    rows.append(prot)
    tids.append(row['target_id'])
db_session.add_all(rows)
db_session.commit()
db_session.remove()
