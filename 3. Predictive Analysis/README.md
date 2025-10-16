# Flight Status Prediction with Multiclass Classfication
# Summary
Based on public 2008 commercial flights data from the United States, this is a multi-class classification task to predict whether a flight will be cancelled, diverted, delayed or on-schedule and is performed by four classification algorithms: logistic regression, tree-based models, naïve bayes and neural networks.
The classification accuracy, ROC-AUC (Area Under Curve) and F scores of these algorithms are compared. Experimental results show that the highest cross-validated classification accuracy belongs to tree-based model random forest with 98% and neural network with almost 96%. In terms of macro-averaged AUC however, tree-based gradient boosting performs best with 0.58 where the score for random forest is 0.53. In terms of micro-F-score, random forest is the second best with 0.92 and neural network is the best with 0.96 where for macro-F-score the best is gradient boosting with 0.30. Tree-based models are suggested for overall classification accuracy where further model tuning and/or higher dimensionality is recommended for better minority class prediction. Several other insights in accordance with prediction results are also reported as conclusions of this study.
# Pre-Processing and Customisation of Data
Standard data cleaning steps are applied including removal of attributes with no predictive value and missing value elimination. 
Only realistic attributes are used for modelling i.e., attributes that are naturally not expected to be known before a flight takes place are eliminated.
Categorical variables are transformed into numerical variables.
# Objective: Business Questions
- What is the level of accuracy we can predict the flight status knowing only pre-flight schedule information?
- Which information is the most effective for this prediction?
- Are there any specific destinations or distances to be avoided or a specific month needing more attention for planning?
# Conclusions
- The level of accuracy we can predict the flight status knowing only pre-flight schedule
information: 98% as per cross-validated accuracy of random forest model with recommendation to further tune the model for better prediction of minority classes.
- The most effective information for this prediction: As per the output of random forest model shown in below figure, scheduled elapsed time, distance and scheduled departure time are the
most important features for predicting the flight status.
- Any specific destinations or distances to be avoided or a period needs to focused on: better insights on this are to be achieved after a better tuned model however, with the initial predictions of random forest model we can already say that delays are expected to peak in January with 97% of the flights where diversions peak in February. In terms of destinations, LAS, PHX, and MDW are the ones that majority of delayed flights are expected to occur.

Despite high accuracy rates, low AUC and F-score results show that in terms of capturing the minority classes no model performs effective enough. Further model tuning and/or more attributes in the dataset are needed to improve the performance of predicting these classes which in this case are cancelled and diverted flights. According to initial predictions, January seems to be the most sensitive period with the highest percentage of delays and certain destinations such as LAS, PHX, MDW are expected to have more delays than others during cold season. Airlines may want to avoid these particular destinations or minimise the flights there in line with the demand in order to provide more timely flights.
