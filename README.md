# Hong Kong SEIHR Model

<div align="center">
	<h2 style="color:#d00000; margin-bottom: 0;">!!! WARNING !!!</h2>
	<p><strong>This project is unfinished and VERY MUCH a work in progress.</strong></p>
	<p><strong>Expect literal code in progress. This is very obviously not finished.</strong></p>
</div>

This project builds a stratified SEIHR-style compartmental model for Hong Kong COVID-19 data using three age groups: children, adults, and elders.

The goal is to study how serious-case and hospitalization data changed across major waves, especially Alpha, the 5th wave, and Omicron, and to compare how effective public health and social measures were against Alpha versus Omicron.



## What's in the repo

- `model/SEIHDR.py` contains the age-stratified SEIHDR model.
- `model/matrixgenerator.py` builds the 3x3 contact matrix used by the model.
- `data/plot.py` loads Hong Kong government data and plots the series.
- `data/HKGov.csv` stores the dataset used for the hospitalization/critical-case analysis.

## Dependencies

Install the Python packages listed in `requirements.txt` before running the scripts.

## Notes

The project uses hospitalization or critical-case data as the `H` compartment and looks at how transmission parameters and effective $R_0$ changed across waves.

## Data

The data is from Hong Kong Official government data. https://data.gov.hk/en-data/dataset/hk-dh-chpsebcddr-novel-infectious-agent/resource/9252c845-3aea-4ea7-abae-b385916106b3
