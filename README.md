# Monta_project

This repository contains the code pertaining to the research conducted with the aim of short-term forecasting of the
electricity spot price of the day-ahead market. In particular, the aim is to obtain a four-day-ahead
forecast of the Spot price of the Production of electricity in the day-ahead market.

Having a longer reliable window of known prices is essential to be able to offer Monta’s charging system users the lowest price for the period during which their car is connected to the charging point (CP). Monta is not an electricity provider, but an intermediary between end-users and utilities companies as well as providers and as such it seeks to charge the customer the lowest possible price to fully charge the car. As depicted in figure 1.2, users will connect their cars to
the CPs and perhaps go to work to their offices, where their interest is not to charge immediately but to charge preferably at the lowest price, and finished the charge when they finish the work day and they need to leave. In reality, what Monta sees in Denmark and other Scandinavian countries is that users do not need to charge every time since they do short distances, or perhaps leave the car for several days connected to their home charger. It is in this situations when smart charging is most interesting and where the four-day ahead forecasts enter.

![monta_charging](Monta%20charging.png)

The project studies how different models: historical averages, classical time series models, recurrent neural networks and probabilistic RNNs forecast the price of electricity, first with a focus in NO1 region and later across Scandinavian countries. The results of this generalization can be defined by the ratio of much improvement do these models achieve with regards to the RMSE of forecasting with hisotrical averages. These can be seen below:


![results_regions](https://file%2B.vscode-resource.vscode-cdn.net/c%3A/Users/ivorr/Documents/GitHub/Monta_project/Results%20of%20models%20across%20regions.png?version%3D1656258017081)



**Notes:**
The research project has been conducted as part of a collaboration between Monta and the
Machine Learning for Smart Mobility section of the Technical University of Denmark.

The project in whole can be seen in the pdf document: *Electricity Price forecasting using Probabilistic Deep Learning*. The code for the different stages of the project can be checked in the folder *Project's notebooks*.

