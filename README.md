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

## Null Hypothesis
- School Time and Travel Time has no affect on social media usage.
- Sleep Time and Sleep Quality has no affect on social media usage.
- Homework and Projects have no affect on social media usage.
- Social Interaction has no affect on social media usage.
  
## Expected Outcome
Insights: I expect the usage of social media to increase when I have less sleep, have lesser social interaction and have longer travel and school time.
## Expected Visualizations
-Time-Series Plots: To track social media usage over time and look for trends.
-Correlation Heatmaps: To see how different factors relate to social media usage.
-Correlation plots: To show the correlation between two selected data columns.
-Regression Models: To quantify how much each factor contributes to usage frequency.

## Insights After EDA
After Exploratory Data Analysis I can say these things about the hypothesis I had at the beginning of the project:
## 1-SCHOOL TIME AND TRAVEL TIME 
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

We can see that relations are significant. We can reject the null hypothesis. School time affects my social media usage.

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

We can see that relations are significant. We can reject the null hypothesis. Travel time affects my social media usage

## 2-SLEEP TIME AND SLEEP QUALITY
-The null hypothesis is that with increased Sleep Time and Sleep Quality my social media usage will not be affected.
-By looking at correlation between sleep time and 2 social media usage data we see that:
![resim](https://github.com/user-attachments/assets/7050345c-dd52-4457-8fe1-d408b70a47c9)
![resim](https://github.com/user-attachments/assets/f0d62a64-6260-40f5-b78c-5196999e166e)

My Instagram Usage doesn't have a statistically significant correlation while my Reddit Usage has a statistically significant negative correlation with sleep time although not that strong. 
If we check the Chi Square and T-Test:

Statistical Tests for Sleep Time vs Instagram Usage
==================================================
Chi-Square Test:
Chi2 statistic: 2.303
P-value: 0.129
T-Test:
T-statistic: 1.941
P-value: 0.061
Interpretation:
- No significant relationships found in any test

Statistical Tests for Sleep Time vs Reddit Usage
==================================================
Chi-Square Test:
Chi2 statistic: 2.378
P-value: 0.123
T-Test:
T-statistic: -3.333
P-value: 0.002
Interpretation:
- T-Test: Significant difference in means found

If we look at these results, since p value is bigger than 0.05 and the other tests gave no significant relation found for Sleep Time vs. Instagram Usage we can say that the null hypothesis can't be rejected for Instagram Usage.
However for Reddit Usage the test is significant and the p value is less than 0.05. So we can say that for Reddit Usage vs Sleep Time we can reject the null hypothesis but generally we can not. Because there is a significant relation between Reddit Usage and Sleep Time (negative). 

-By looking at correlation between sleep quality and 2 social media usage data we see that:
![resim](https://github.com/user-attachments/assets/0d6e5855-52b6-4ce2-9e6b-0d0d82ee09c8)
![resim](https://github.com/user-attachments/assets/e83ecf25-fd53-4b99-bf92-1279f96e6a07)

My Instagram and Reddit Usage doesn't have a significant relation with Sleep Quality although for Reddit Usage the p value is really close to 0.05 (it is 0.055). If we look at chi square and t-test
results:

Statistical Tests for Sleep Quality vs Instagram Usage
==================================================
Chi-Square Test:
Chi2 statistic: 0.232
P-value: 0.630
T-Test:
T-statistic: 0.702
P-value: 0.487
Interpretation:
- No significant relationships found in any test

Statistical Tests for Sleep Quality vs Reddit Usage
==================================================
Chi-Square Test:
Chi2 statistic: 5.322
P-value: 0.021
T-Test:
T-statistic: -2.099
P-value: 0.044
Interpretation:
- Chi-Square: Significant relationship found
- T-Test: Significant difference in means found

If we look at the results we can see that for Instagram Usage again there is no significant result. But for Reddit Usage there is a significant relation. The correlation gave no significant relation for Sleep Quality vs Reddit Usage with a very close p value to 0.05 and the Chi-Square and T-Test gave there is significant relation between them. Since there is a different interpretation between different methods we can say that we can not reject null hypothesis. 

## 3-SOCIAL INTERACTION

The null hypothesis was that Social Interaction has no affect on social media usage. If we look at correlations between Social Interaction vs Reddit and Instagram Usage:
![resim](https://github.com/user-attachments/assets/d7b93ed5-3c33-4d14-9b67-02aec5697963)
![resim](https://github.com/user-attachments/assets/9bb08b66-6c53-4e86-961c-ffc75e72081c)

There is no significant relation between Social Interaction and Instagram but for Reddit Usage there is a significant positive correlation. If we look at chi square and t-test:

Statistical Tests for Social Interaction vs Instagram Usage
==================================================
Chi-Square Test:
Chi2 statistic: 1.802
P-value: 0.179
T-Test:
T-statistic: -2.022
P-value: 0.051
Interpretation:
- No significant relationships found in any test

Statistical Tests for Social Interaction vs Reddit Usage
==================================================
Chi-Square Test:
Chi2 statistic: 3.263
P-value: 0.071
T-Test:
T-statistic: 2.860
P-value: 0.007
Interpretation:
- T-Test: Significant difference in means found

If we look at the results we can say that for Instagram Usage there is no significant relation with Social Interaction. For Reddit, here we also have a significant relation so we can reject the null hypothesis for Reddit Usage vs Social Interaction. Generally we fail to reject it. 

## 4-HOMEWORK AND PROJECTS
The null hypothesis was that Homework and Projects have no affect on social media usage. If we look at correlations between HW/Projects vs Reddit and Instagram Usage:
![resim](https://github.com/user-attachments/assets/6729e3d5-1ead-4979-b7ab-4f3c3bdad9db)
![resim](https://github.com/user-attachments/assets/1e63e5d1-c10a-4902-b239-807e1e05cce7)

We see that for both there is no significant relation. There is no chi-square and t-test for HW/Projects because the only values they have is either 1 or 0. Instead we have a boxplot visualizations for both:
![resim](https://github.com/user-attachments/assets/349a90e1-06f6-47fb-b589-3745482400d9)
![resim](https://github.com/user-attachments/assets/8c2abc6c-3d17-4bcd-a69f-c510c6827ea6)

The mean of Reddit Usage is higher when a homework/project exists. The mean of Instagram usage is lower when a homework/project exists. Logically, if we look at boxplots we see that Reddit Usage is more when there is a homework or project but of course since I wasn't able to make more data interpretations we can not say that we can reject the null hypothesis. 

## EDA END:
-Logically we can see a pattern here. When I am at school I use Reddit more than Instagram. We can understand it by looking at school time and travel time datas. Also since I sleep less when I go to school my Reddit Usage has a significant relation with Sleep Time and Sleep Quality but of course these are my thoughts. Also since I have more social interaction at school Reddit Usage is also being affected by it. Similarly when I have more homeworks or projects I spend more time at school so it is also affecting Reddit Usage.


## MACHINE LEARNING:
-Here I tried to predict another social media application I use: Youtube. Using the same data set I had for Reddit and Instagram I tried predicting the screen time of my Youtube usage using Linear Regression, Decision Tree and Random Forest, Support Vector Regression, KNN Regression and Gradient Boosting seperately.

![resim](https://github.com/user-attachments/assets/a1e64d7f-1350-49c8-a473-8099fc370c24)
![resim](https://github.com/user-attachments/assets/7867499b-0ba5-4cdb-93d2-e8dcd12dfe4c)
![resim](https://github.com/user-attachments/assets/1f071a8b-c8df-434a-a37c-90c4653c4a38)
![resim](https://github.com/user-attachments/assets/c0d856b2-427c-47a4-80fd-bf206a78f6e2)
![resim](https://github.com/user-attachments/assets/0662b66c-b274-4030-a431-4f795f9a1899)
![resim](https://github.com/user-attachments/assets/0f9476b2-6f97-49ca-be3b-2340aca3f24e)


-Here my Youtube Usage is best preditected by Support Vector Regression since it has the lowest error results for RMSE and MAE and has the highest R square value. It achieves the best balance between low error rates and variance explanation. KNN and Decision Tree also show similar results but they have higher error rates and lower R square value. 
-However none of the models explain the data significantly, so from here we can say that maybe Youtube data is affected by other data sets I dont have in my data or the relationship between Youtube usage and the indicators in my data is more complex than I think.

-----------------------------------------------------------------------------------------

-Here I tried to predict Instagram Usage and Reddit Usage by using Linear Regression, Decision Tree, Random Forest, Support Vector Regression, KNN Regression and Gradient Boost as well:
Instagram:

![resim](https://github.com/user-attachments/assets/e5f6d840-afe5-48b0-869e-9b87611e1dfd)
![resim](https://github.com/user-attachments/assets/e5a5776a-b39b-457a-a3db-dfd6f84b0da2)
![resim](https://github.com/user-attachments/assets/33198868-28e5-4380-8cd7-ba855c94e5ef)
![resim](https://github.com/user-attachments/assets/0a49b9d5-1b34-4026-baba-7eaa90d2ddc0)
![resim](https://github.com/user-attachments/assets/2569a540-81d7-486e-bd13-a3a5096d063e)
![resim](https://github.com/user-attachments/assets/b1e016fd-dba0-474b-8e41-c9a7b858c9cb)


-Here if we look at the data we can see that Support Vector Regression is the best model for predicting Instagram Usage with the highest R squared and lowest errors results. It captures nonlinear patterns well and outperforms other models. Random Forest and KNN also performs strong, but not as much as SVR.
-Compared to the results for YouTube Usage the models for Instagram Usage performed better likely because the data was first gathered to make comparision between Reddit and Instagram Usage and Youtube usage was not considered in the early stages of project planning. 



Reddit:

![resim](https://github.com/user-attachments/assets/e67e8b25-f1a4-4d8e-b266-0bb6432703e7)
![resim](https://github.com/user-attachments/assets/89da84c4-2223-4732-8f6f-5fbb272033ea)
![resim](https://github.com/user-attachments/assets/2cbe01cf-24fe-4335-b082-2083ffcd629a)
![resim](https://github.com/user-attachments/assets/2ea5cfd4-909b-49ad-8806-32f599d7c27e)
![resim](https://github.com/user-attachments/assets/01188e6a-d8a7-42b8-979f-5f1166f751ae)
![resim](https://github.com/user-attachments/assets/78769eaf-f4f7-4855-b2b5-192fd472037e)


-Here if we look at the data we can see that Random Forest Regression is the best model for Reddit Usage with the highest R squared and lowest error results. Decision Tree Regression also provides strong results with simpler structure. Since Linear Regression and SVR underperform we can say that Reddit usage is influenced by nonlinear factors most of the time.
-Out of all the result we had, the R square for Random Forest with 0.85 was the highest one. Suggesting that the data set I have is affective the most when we try to predict my Reddit Usage.










