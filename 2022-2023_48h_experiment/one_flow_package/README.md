# Image warping using model data and assessing the accuracy of such predictions.

Author :      Anna Telegina

Copyright :   (c) CIRFA 2024

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage example](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)


## Description

The script processes pairs of SAR images for the experiment of quality assessment of the sea ice condition forecasting using model drift data. It begins by pairing SAR images based on their timestamps and prepares them for analysis. The main steps include:

-Preparing SAR Pairs: Collects and pairs SAR images based on the time difference between their captures (SAR1 and SAR2 in each pair).
-SAR Drift Retrieval: Calculates the drift between SAR image pairs using feature tracking and pattern matching techniques.
-Cumulative Model Drift Calculation: Integrates hourly model data to compute cumulative drift over the period between the two SAR images.
-Warping SAR Images: Utilizes calculated drift fields to warp the first SAR image, projecting its future state for the moment at SAR2 retrivial.
-Quality Assessment: Compares the warped SAR image (SAR1 predicted) with the SAR2 and calculates distortion parameters to assess the quality of the warping process.

## Installation

Running with Docker
```
add 
```
Updating configuration file

config.py
The config.py file contains user-defined parameters. Here its structure and parameters that might be updated:

-Directories and File Paths: Update the paths to the directories containing your SAR image files. You can set path_to_HH_files, path_to_HV_files, safe_folder, output_folder, and input_folder according to your file structure.

-Regular Expressions: These regular expressions are used for matching file names. You may need to modify them to match your file naming conventions if they differ.

-Grid Configuration: This section loads grid information from the barent_grid.npz file. Ensure that the path to this file is correct.

-Filtering Parameters: Set the values for filtering parameters like hessian, neighbors, disp_legend_min, and disp_legend_max as needed for your project.

## Usage

```
add using an example of one pair 
```
