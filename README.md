# My Social Media Usage
## PROJECT IDEA and MOTIVATION

In this project my aim is to analyze the frequency of my social media usage and how various factors throughout the day influence it. The goal is to understand whether there is a real relation between the social media usage and my regular daily routine. By analyzing factors such as school attendance, exam and project schedules, travel time, sleep time and quality, amount of social interaction I had in a day, I hope to uncover patterns that could help me understand the reasoning of my time spent on social media. 

##  DATASET DESCRIPTION 
- **Date** : Data entry day
- **Having a Homework or a Project Deadline in Upcoming three days**: YES or NO, will be marked by me.
- **Amount of Time In School**: In terms of hours, I will be using Google Maps to check how many hourse I stayed in school.
- **Travel Time**: In terms of minutes or hours, will be using Google Maps to check it. 
- **Sleep Time**: In terms of hours, will be using a smartwatch to collect the data.
- **Sleep Quality**: 1-10, I will be using the same smartwatch and same app to collect the data.
- **Social Interaction**: In terms of numbers, it will start with 0 and with every non-family member who I talk more than 3 minutes I will add one to it.
- **Reddit Usage**: Will be collected from the settings part of my smart phone
- **Instagram Usage**: Will be collected from the settings part of my smart phone
# DATA COLLECTION
- Data will be collected for 30 days. (14th March to 13th April)
- Sources:
    - Smartwatch application for sleep time and quality
    - Google Maps for school time and travel time
    - Settings part of my smart phone to gather hourly usage of two social media I use (Reddit and Instagram)
- To ensure consistency and minimize bias:
    - Information will be recorded daily
    - Data will be systematically organized and accessible.
    - Confounding variables and outliers will be carefully considered to ensure the accuracy and reliability of the analysis.

#### **Data Preparation and Analysis** 
- At the end of the data collection period: 
	- Data will be reviewed for completeness and consistency. 
- **Exploratory Data Analysis (EDA):** 
	- Trends will be identified using statistical methods and visualization techniques. 
- **Regression Analysis:** 
	- Statistical methods will be applied to investigate the impact of various factors such as departure time and weather on travel duration and stress levels.

## Hypothesis
I hypothesize that my social media usage is more frequent on days with:
- More school time and travel time
- Less sleep and lower sleep quality
- More stressful academic events, such as exams and project deadlines.
- Lower social interaction.
  
## Expected Outcome
Insights: I expect the usage of social media to increase when I have less sleep, have lesser social interaction and have longer travel and school time.
## Expected Visualizations
-Time-Series Plots: To track social media usage over time and look for trends.
-Correlation Heatmaps: To see how different factors relate to social media usage.
-Correlation plots: To show the correlation between two selected data columns.
-Regression Models: To quantify how much each factor contributes to usage frequency.

## Insights After EDA
After Exploratory Data Analysis I can say these things about the hypothesis I had at the beginning of the project:
1-School Time and Travel Time
-The null hypothesis is that with increased School and Travel time my social media usage will not be affected.
-By looking at correlation between school time and 2 social media usage data we see that:
![resim](https://github.com/user-attachments/assets/be19d1e6-761a-4471-9277-8de6d6ac1e34)
![resim](https://github.com/user-attachments/assets/a4245f56-c673-4d0b-8ada-814a181b3d02)

My Instagram Usage has a statistically significant negative correlation while my Reddit usage has a has a statistically
significant positive correlation with my school time. p value is smaller than 0.05 so we can say that the null hypothesis is wrong
and there is a significant relation between my social media usage and school time. Also if we check the Chi-Square Test and the T-Test:

Statistical Tests for School Time vs Instagram Usage
==================================================
Chi-Square Test:
Chi2 statistic: 10.362
P-value: 0.001
T-Test:
T-statistic: -4.529
P-value: 0.000
Interpretation:
- Chi-Square: Significant relationship found
- T-Test: Significant difference in means found

Statistical Tests for School Time vs Reddit Usage
==================================================
Chi-Square Test:
Chi2 statistic: 20.867
P-value: 0.000
T-Test:
T-statistic: 6.805
P-value: 0.000
Interpretation:
- Chi-Square: Significant relationship found
- T-Test: Significant difference in means found

We can see that relations are significant. We can reject the null hypothesis.

-By looking at correlation between travel time and 2 social media usage data we see that:
![resim](https://github.com/user-attachments/assets/6abecd6e-3f97-47b8-9d97-60c3e61da813)
![resim](https://github.com/user-attachments/assets/9f280813-2eb2-4dd5-ba40-3d4ddb7d923e)

My Instagram Usage has a statistically significant negative correlation while my Reddit usage has a has a statistically
significant positive correlation with my travel time. p value is smaller than 0.05 so we can say that the null hypothesis is wrong
and there is a significant relation between my social media usage and travel time. Also if we check the Chi-Square Test and the T-Test:

Statistical Tests for Travel Time vs Instagram Usage
==================================================
Chi-Square Test:
Chi2 statistic: 10.362
P-value: 0.001
T-Test:
T-statistic: -5.073
P-value: 0.000
Interpretation:
- Chi-Square: Significant relationship found
- T-Test: Significant difference in means found

Statistical Tests for Travel Time vs Reddit Usage
==================================================
Chi-Square Test:
Chi2 statistic: 15.125
P-value: 0.000
T-Test:
T-statistic: 4.096
P-value: 0.000
Interpretation:
- Chi-Square: Significant relationship found
- T-Test: Significant difference in means found



