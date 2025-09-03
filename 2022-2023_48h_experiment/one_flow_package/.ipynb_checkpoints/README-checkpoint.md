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

Overview

- Repository "sea_ice_docker_upgrade" contains everything needed to build a Docker image and run JupyterLab for SAR drift experiments:

- Repository name: sea_ice_docker_upgrade

- Dockerfile: sea_ice_docker_upgrade/Dockerfile

- Docker image name: sar_forecast_experiment

- Default container name: seaice_experiment (In my case main experiment container included the period of time SAR images were downloaded for: 2022-2023_48h_experiment.)


1. Install Docker Desktop (Windows)

- Download and install Docker Desktop.

- Enable WSL 2 and install the Linux kernel update package if prompted.

- In Docker Desktop → Settings → Resources → File Sharing, grant access to your GitHub and data folders.

- Restart your computer.

2. Build the Docker Image

- Open PowerShell in the folder containing the Dockerfile (sea_ice_docker_upgrade) and run:

    docker build -t sar_forecast_experiment .


This command:

- Uses the Dockerfile to create a reproducible JupyterLab image.

- Installs nansat, gdal, xarray, scikit-image, and other dependencies.

- Sets JupyterLab (start-notebook.sh) as the default entry point.

3. Run a Container


docker run -d --name 2022-2023_48h_experiment -p 8888:8888 -v "C:\Users\<username>\OneDrive - UiT Office 365\Documents\GitHub\SAR_forecast_experiment:/home/jovyan/work" -v "C:\Users\<username>\OneDrive - UiT Office 365\Documents\data_for_experiments:/home/jovyan/data" sar_forecast_experiment

Where:

Start a container in the background (detached)

    docker run -d
    
Give the container a memorable name so you can start/stop it later

    --name 2022-2023_48h_experiment
    
Map host port 8888 -> container port 8888 (Jupyter listens on 8888 in the container)

  -p 8888:8888
  
Mount your local GitHub repo into the container at /home/jovyan/work (Jupyter’s working area)

  -v "C:\Users\<username>\OneDrive - UiT Office 365\Documents\GitHub\SAR_forecast_experiment:/home/jovyan/work" 
  
Mount your local data folder into the container at /home/jovyan/data

  -v "C:\Users\<username>\OneDrive - UiT Office 365\Documents\data_for_experiments:/home/jovyan/data" 
  
Use the image you built earlier

  sar_forecast_experiment


4. Access JupyterLab

Fetch the login URL and token:

docker logs 2022-2023_48h_experiment | Select-String -Pattern token

Open the URL or go to http://localhost:8888/lab and paste the token.

5. Start vs Logs
Command	Purpose
docker start seaice_experiment	Start container silently.
docker start -ai seaice_experiment	Start and attach (shows logs)-> you can use a URL from here to open Jupyter Lab. 
docker logs seaice_experiment	View previous logs, including the Jupyter token.


Advantages of Using Docker

Reproducibility: Every run uses the same Python, library versions, and configuration.

Easy collaboration: Share your Docker image or Dockerfile, and colleagues can recreate your environment exactly.

Cross-machine portability: Run the same container setup on Windows, macOS, or Linux without reinstallation.

Version control integration: The /work folder is linked to your GitHub repository for code sync.

Fast onboarding: Colleagues only need Docker to start experimenting.

Experiment isolation: Create multiple named containers (e.g., 2022-2023_48h_experiment) for different analyses without affecting each other


### Updating configuration file

config.py
The config.py file contains user-defined parameters. Here its structure and parameters that might be updated:

- Directories and File Paths: Update the paths to the directories containing your SAR image files. You can set path_to_HH_files, path_to_HV_files, safe_folder, output_folder, and input_folder according to your file structure.

- Regular Expressions: These regular expressions are used for matching file names. You may need to modify them to match your file naming conventions if they differ.

- Grid Configuration: This section loads grid information from the barent_grid.npz file. Ensure that the path to this file is correct.

- Filtering Parameters: Set the values for filtering parameters like hessian, neighbors, disp_legend_min, and disp_legend_max as needed for your project.

## Usage

```
add using an example of one pair 
```
