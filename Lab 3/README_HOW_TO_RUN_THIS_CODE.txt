Because some peer reviewing groups were unable to open Jupyter Notebook at all for earlier assignments, here are very detailed instructions to run the code.

All code is tested with Python 3.7. All complex code is abstracted away in 'helper.py' files, which are called and executed in Jupyter (Lab) notebooks.

Each subtask (sampling, sketching, etc.) has it's own, clearly labeled folder. Open and run the notebooks in those folders to make sure the output is similar to the results in the report, and inspect to code in the helper.py files.

If you are unable to run the code, please download and install a fresh copy of Anaconda3 2019.03 (https://www.anaconda.com/distribution/), install the following packages (using `pip install [package-name]` in Anaconda Prompt):

 - mmh3 (https://pypi.org/project/mmh3/)
 - imblearn (https://imbalanced-learn.readthedocs.io/en/stable/install.html)
 
 
and execute `jupyter lab` in Anaconda Prompt.


Please note that the source data IS NOT INCLUDED because it is too large for GitHub. Download instructions are included in the `data` folder. Essentially, download https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/capture20110811.pcap.netflow.labeled into the `data` folder.


Thanks, have fun!