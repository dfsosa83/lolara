# Project Title:

## LolaRa: Anticipating Extreme Patterns in Euro/USD Data for Strategic Trading
(Ra is the ancient Egyptian god of the sun and creation. He is considered one of the most important deities. A model named LolaRa might imply illumination, 
enlightenment, or creation)

### Project Description:
The Odysmerc project is dedicated to analyzing Euro/USD data to identify extreme patterns and anticipate uncommon price fluctuations. By leveraging data analysis and strategic trading techniques, the project aims to develop a methodology for earning money through informed trading decisions.

### Features:

Data analysis to identify extreme patterns and unusual price movements
Strategy development for trading based on anticipated market conditions
Testing and validation of trading strategies to optimize earning potential


### Getting Started:
- Clone the repository to your local machine
- Install necessary dependencies using requirements.txt
- Explore the dataset provided in the project
- Run the analysis scripts to identify extreme patterns
- Develop and test trading strategies based on the findings

### Data Sources:

- Euro/USD historical data 

### Methodology:
- Outline the approach taken to identify extreme patterns and develop trading strategies

# ########################################### To env - Methodology ################################################################
- Make sure your environment.yml file looks something like this:

name: Odysmerc
channels:
  - defaults
  - conda-forge
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - tensorflow
  - pip
  - pip:
      - keras
      - some-other-pip-package


## Create the Environment:
- conda env create --file environment.yml

## Activate and Verify:
1. conda activate Odysmerc 
OR
2. source activate Odysmerc

- python -c "import tensorflow as tf; print(tf.__version__)"
- python -c "import keras; print(keras.__version__)"





- Update the Conda Environment:

Open your terminal and navigate to the directory containing your environment.yml file. Then run the following command to update your environment:

- conda env update --name lolara --file environment.yml --prune

- Remove the Environment:
- conda env remove --name lolara
- conda env list


## Activate Your Environment:

conda activate Odysmerc

## Some Explanations in yml file:

The channels section in a Conda environment.yml file specifies the repositories (also known as channels) from which Conda should fetch the packages you need. Conda channels are sources where packages are stored and managed.

### Why Use Channels?

- Default Channel: By default, Conda packages are fetched from the defaults channel, which is maintained by Anaconda, Inc.

- Community Contributions: Some packages are not available in the defaults channel but are available in other channels like conda-forge, which is a community-driven collection of packages.

- Specific Needs: Different channels can have different versions of packages or packages built with different dependencies (e.g., using different compilers).

Common Channels

- defaults: The default channel provided by Anaconda, Inc.

- conda-forge: A community-driven channel that has a vast collection of packages, often with more current versions than the defaults channel.

- bioconda: A channel specifically for bioinformatics software.

- anaconda: This is for Anaconda's own packages, which might be slightly different from defaults.


