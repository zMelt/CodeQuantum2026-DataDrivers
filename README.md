## Inspiration

This project was inspired by the desire to improve F1 team/driver performance and predictive interactability to further drive fan engagement and outreach.

## What it does

Utilizing bootstrapping and logistic regression, this project trains 25-500 models to provide predictions for top 10 placements, podium placements, and winners. 

## How we built it
We initially prototypes the necessary functions and graphs using a Google Colab Notebook. Then, we compiled the final draft into a Python script that uses Streamlit and Plotly as a front end.

## Challenges we ran into

Creating a Monte Carlo simulation proved to be difficult due to it producing outliers in our predictions. We also did not realize beforehand that our model at the time did not account for races taking place on different tracks throughout the season. Because of this, we pivoted to a bootstrap method utilizing logistic regression to provide more accurate predictions. Learning Streamlit and integrating it into our project on the fly proved to be a challenge as well given the time constraints.

## Accomplishments that we're proud of

Learning the different data science techniques used for predictive modeling and applying them to the provided F1 dataset proved to be a difficult but rewarding task after. 

## What we learned

We gained extensive knowledge on construction of Monte Carlo simulations and bootstrapping, utilizing them for training models and resampling with replacement, and Streamlit. 

## What's next for Data Drivers

In the future, Data Drivers plans to add functionality for directly comparing drivers and finding head to head values, and incorporating track design with features such as turns and circuit rotation to provide more detailed predictions.  Utilize PennyLane to begin scaling quantumly by training on historical telemetry to predict overtaking success.  We also aim to enhance readability and UI navigation for non-technical audiences to improve fan engagement and outreach. 
