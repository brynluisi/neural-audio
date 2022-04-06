# neural-audio

### Running the Models 

The three models: Pure_FIRNN, FIRNN, and LSTM can be run by running their respective 
jupyter notebook in under the `notebooks` directory. The input data should be placed under the neural-audio
root directory. 

### Visualizations

Visualizations for models can be found in the notebooks with "visualization" in the name. 

### Saved Models 

The state_dict of the trained models are in the `models` directory and can be loaded in either 
the model or visualization notebooks. The models used for the visualization notebooks are hard-coded
and should run without error. 

### Output

#### Audio 
The sound examples for the output audio for the wav files are under the `output/audio` directory. 
The input and target audio is under the `/data/train` directory. 

#### Images 

Images produced by the visualization notebooks are saved in `output/images`. 