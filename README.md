# TDT4173 – Modern Machine Learning in Practice

This repository contains my project for the course **TDT4173 Modern Machine Learning in Practice**.

## Project Description
The task was to **forecast the cumulative weight of received raw materials** at a factory in Spain.  
The forecasts were made **per raw material**, using:

- Historical receival data  
- Purchase order information  

## Approach
I performed **extensive feature engineering**.

I experimented with different modeling strategies and found that:

- **Training one LGBM model per raw material** produced the best results.

## Results
The final solution achieved:

- **14th place out of 304** on the Kaggle leaderboard  
- Outperformed **9 out of 10 virtual teams**