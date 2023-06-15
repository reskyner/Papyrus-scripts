# Kinase active vs. inactive classification
## Example of ustilising postgres Papyrus implementation and classification example
### Presented in a poster at Sheffield Cheminformatics conference 2023
---

### Files:   
---   
- ``get_papyrus_data.py:`` code dump of pulling data from Papyrus postgres (includes sqlalchemy queries and pre-processing [e.g. calculating rdkit descriptors])   
- ``classifier_pipeline.py:`` sll functions needed to build classifier models for datasets generated with ``get_papyrus_data.py``   
- ``models.py:`` sqlalchemy models used in ``get_papyrus_data.py``. This is a copy of the models from ``../database/models.py``, but is included here so that if the Payprus postgres database is created from the Zenodo data dump, there is no need to install the whole package, you can just use the scripts in this directory in isolation.   
- ``session.py:`` function to get database connection.
` ``sheffield_poster.pdf``: A copy of the poster presented at Sheffield 