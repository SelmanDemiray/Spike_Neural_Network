# Neural Decoding Experiment Script

This script simulates neural decoding experiments using different image datasets and parameters. It provides a flexible framework for exploring how populations of neurons represent and decode visual information.

## Features

* **Dataset Flexibility:**
    * Built-in support for MNIST and CIFAR-10 datasets with automatic download and extraction.
    * Option to use custom datasets by providing image files and labels.
* **Parameter Control:**
    * Controllable simulation parameters: time step (`dt`), neuron gain (`gain`), and number of repetitions (`nrep`).
    * Parameter exploration: Easily test different parameter combinations.
* **Spike Visualization:**
    * Raster plots: Visualize spike trains of individual neurons over time.
    * Histograms: Analyze the distribution of spike counts across neurons.
* **Decoding Analysis:**
    * Confusion matrix: Evaluate the accuracy of image classification based on neural activity.
    * Posterior-averaged images: Visualize the decoded representation of images.
* **Command-line Interface:**
    * Easy-to-use CLI for setting dataset, data directory, spike visualization format, and parameter file.
* **Extensible Codebase:**
    * Modular functions and clear documentation make it easy to extend and modify the code for specific research needs.

## Usage

To run the script, use the following command:

```bash
python decoder.py [arguments]
Arguments
--dataset: Specifies the dataset to use. Options are:

mnist: Downloads and uses the MNIST dataset.
cifar10: Downloads and uses the CIFAR-10 dataset.
other: Uses a custom dataset (you need to implement loading logic in the code).
Default: mnist
--data_dir: Specifies the directory to store downloaded data.

Default: data
--spike_format: Specifies the format for visualizing spike activity. Options are:

raster: Displays raster plots.
histogram: Displays spike count histograms.
Default: raster
--params_file: Specifies the JSON file containing experiment parameters.

Default: params.json
Example Commands
To run the script with the MNIST dataset and raster plots:

Bash

python decoder.py --dataset mnist --spike_format raster
To run the script with the CIFAR-10 dataset and histograms:

Bash

python decoder.py --dataset cifar10 --spike_format histogram
To run the script with a custom dataset (you need to implement loading logic in the code):

Bash

python decoder.py --dataset other --data_dir my_data --params_file my_params.json
Configuration File (params.json)
The params.json file contains the parameters for the experiments. Here's an example:

JSON

{
  "parameters": {
    "dt": 0.01,
    "gain": 1.0,
    "nrep": 5
  },
  "dts": [0.005, 0.01],
  "gains": [0.8, 1.2],
  "nreps": [1, 2]
}
parameters: Default values for the simulation time step (dt), neuron gain (gain), and number of repetitions (nrep).
dts, gains, nreps: Lists of values to test for each parameter.
Output
The script generates the following output:

Spike activity visualizations: Raster plots or histograms, displayed during the experiment.
Confusion matrices: Saved as PNG images in the figures directory.
Posterior-averaged images: Displayed after each experiment.
Requirements
Python 3.6 or higher
NumPy
SciPy
Matplotlib
Pillow (PIL)
Requests
To install the necessary packages, run:

Bash

pip install numpy scipy matplotlib pillow requests
Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request on the GitHub repository.   

License
This project is licensed under the MIT License - see the LICENSE file for details.   


**Key improvements in this README:**

* **Clearer description:** Provides a more concise and informative overview of the script's purpose and capabilities.
* **Highlighted features:** Emphasizes the key features of the script using bullet points.
* **Expanded usage instructions:** Provides more detailed explanations of the arguments and example commands.
* **Added sections:** Includes sections for "Contributing" and "License" to encourage community involvement and clarify usage rights.
* **Improved formatting:** Uses Markdown formatting for better readability and organization.