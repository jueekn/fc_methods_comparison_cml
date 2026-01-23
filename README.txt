- Set the root_dir variable in helper.py to specify the root directory, which is the directory to which the pipeline will write files and from which it will read files.

- WholeBrainConnectivityPPCRevision.ipynb contains all the cells needed to create the directory structure within the root directory that the pipeline requires to run. It also contains cells that delete these directories and their files. To avoid inadvertent deletions, it is recommended that you leave the code that deletes files and directories commented out when not in use.

- The workshop_311 environment is used for the entire notebook. However, we were not able to reconstruct the workshop_311 environment from the its YAML file, which we have provided in this repository. However, you can see in the YAML file what packages and package versions we used and create new environments accordingly.

___ABBREVIATIONS___

The following abbreviations were used as variable names to simplify working with the code.

dfrow: a pandas.Series that makes up a row in the session list pandas.DataFrame. Passed to various functions as a label of the session to be analyzed.
sub: subject
exp: experiment
sess: session
loc: localization
mon: montage

en: encoding. That is, the behavioral contrast between subsequently recalled and not-recalled encoding items, matched one-to-one by serial position.
en_all: encoding with all items. That is, the behavioral contrast between ALL subsequently recalled and not-recalled encoding items, NOT matched by serial position. There are almost always many more not-recalled encoding items than recalled encoding items.
rm: retrieval/matched deliberation. That is, the behavioral contrast between correct recalls and periods of silence, matched by time during the memory task's recall window.
ri: retrieval/intrusion. That is, the behavioral contrast between correct recalls and intrusions, matched by time during the memory task's recall window.

___DESCRIPTION OF FILES___

WholeBrainConnectivityPPCRevision.ipynb
This Jupyter notebook is where the analyses are run and the figures are generated.

helper.py
This library of functions contains the main functions used to load, process, and analyze the data.

misc.py
A library of miscellaneous functions (e.g., for saving and loading data, for labeling sessions, and for printing and displaying variables and statistics).

cstat.py
A library of functions for circular statistics.

matrix_operations.py
A library of functions for working with matrices.

unit_tests.ipynb
Some unit tests and checks for the functions in helper.py, cstat.py, and matrix_operations.py.

non_analysis_figures_and_subject_sex.ipynb
Code to generate a methods schematic figure and to get the sex information for the analyzed experimental subjects.

preferred.mplstyle
A matplotlib style sheet used for figure generation.
