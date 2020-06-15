import helper.visualization as viz
import io
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import os

data = []
preprocessed_data = None
transformed_data = None
# with tempfile.TemporaryFile('w') as summary_file:
for subdir, dirs, files in os.walk('.\data'):
    for file in files:
        curr_file_name = os.path.join(subdir, file)
        with io.open(curr_file_name) as curr_file:
            data.extend(curr_file.readlines())
preprocessed_data = (line.replace("\n", "") for line in data)
transformed_data = (line.split(",") for line in preprocessed_data)

aggregated_data = pd.DataFrame(transformed_data, columns=("method", "N","µ","NMI"), dtype=np.float64)
aggregated_data

viz.draw_plots(aggregated_data) 
plt.show()

