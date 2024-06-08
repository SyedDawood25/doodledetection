# Doodle Detection Using Neural Networks

## Description

Doodle Detection Using Neural Networks is an ML based project that is designed to detect hand drawn doodles during realtime.

# Contents

To run the GUI follow the steps below:

1. Install Dependencies:
   - Open terminal
   - Type "pip install -r requirements.txt"
   - Go to the following link and download the trained model: https://drive.google.com/file/d/1wR17f8u2oqWmVE8z0twx6WiJpA9ZOwvc/view?usp=sharing
   - After downloading the model, move the file to frontend/trained_nn/
2. Navigate to "./frontend/main.py" and run the file
   - Or alternatively open terminal and type " py './frontend/main.py' "

To checkout the preparation of our model

- Navigate to ./MLP_Training.ipynb

To checkout the visualizations of the dataset used for training

1. Make sure to install Anaconda:

   1. Anaconda Distribution: https://www.anaconda.com/download
   2. Miniconda: https://docs.anaconda.com/free/miniconda/

2. Create a virtual environment using conda:

   1. Open terminal and type: "conda create --name myenv"

3. Activate the virtual environment using conda:

   1. Open terminal and type: "conda activate myenv"

4. Install the following dependencies:

   1. Open terminal and type: "conda install -c tmap tmap"
   2. Then install other requirements: "pip install numpy pandas matplotlib scikit-learn faerun umap-learn"

5. Navigate to "./data_visualization.ipynb" and run the cells to generate HTML files for TMAP and UMAP named as "doodlesTMAP.html" and "doodlesUMAP.html".
