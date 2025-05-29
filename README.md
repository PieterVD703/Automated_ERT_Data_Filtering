# Automated_ERT_Data_Filtering
This repository provides a way to automatically filter en visualise data from Electrical Resistivity Tomography (ERT). The data has to be stored in a CSV format. This code was developed for an academic bachelor's project and aims to streamline the data processing.

## Features
- Automated filtering: ability to use a moving window median filter, and/or Butterworth filtering.
- Data visualisation: ability to generate plots to compare filter effects
- configurable global settings: the config.py file changes to filter paramaters globally
- modular design: the organisation of distinct modules and helper functions promotes readability and maintenance.

## Structure
```
Automated_ERT_Data_Filtering/
├── Butterworth_explanation.py   # Demonstrates the effect of Butterworth filtering
├── Filter.py                    # Core filtering functions
├── Plotting.py                  # Functions for data visualization
├── settings/                    # Configuration files for filter parameters
├── data/                        # Directory to store input CSV data files
├── Figuren/                     # Output directory for generated figures
├── Help/                        # Documentation and usage guides
└── README.md                    # Project overview and instructions

```

## Prerequisites
Ensure you haev Python 3.x and the following packages:
```
pip install numpy pandas matplotlib scipy
```
### Usage
1. Prepare data: place the CSV data in the data/ directory. Make sure the csv has a header and seperate data columns
2. Settings: configure the filter paramaters to your needs in the config.py file. Specification of data points happens when initiating the Filter class
3. (Optional) Visualise: you can check your parameters by visualising a data point
4. Run and export: with Filter.py you can export your filtered datapoints to a CSV file.

## Acknowledgements
This project was developed as a bachelor's thesis at UGent.
