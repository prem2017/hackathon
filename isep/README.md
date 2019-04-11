
# Weather parameters prediction from time-series historical-weather data

## Getting started
1. Clone this repository
2. Install [Miniconda](https://conda.io/en/latest/miniconda.html) if you do not have it already. Follow one of the two following steps:
   #### One
    1. Go to the repo-project directory and run
        - $ `conda env create environment.yml`
    2. To activat the environment run
        - $ `source activate torch_v1`

   #### Second
    1. Go to the repo-project directory and run
        - $ `pip install -r requirements.txt`

    **Note**: The second action will override and upgrade the pre-existing libraries. It is recommended to create a separate python `environment` if you have project dependent on old versions. Like `pytorch 0.4` etc.




Agriculture is a field that is very much affected by climatic events: **heat**, **cold**, **frost**, **rain**, **hail**, **wind**.

## These weather events will:

1. Impact crops by causing crop losses, damage to fruits, sometimes for several years (impact of hail on vines, can be felt the following year, with fewer buds)
2. Impact farming operations: the wind limits the possibilities of treatment, rain can make the soil impassable for agricultural machinery.

Note: Read more about the task [here](./Technical_Challenge.pdf)



## DATA

Note: Raw data which includes observation, forcast, aggregation of
forcast is not included in the commit as parsed data from raw is present in data directory.


## Report

The report of this hackathon can be found in [here](./Weather_Prediction_Report.pdf)


## TODOs:
1. Acquire more data to test effectivness of architecture.
2. Tweaking of architecture for better learning.
3. Bring the use of forcast data as well for the forcast task.