# Sales Prediction Project
## Author Hiba Busttami
## Project Overview
Project Overview: Sales Prediction for Multi-Outlet Retail Items
This project aims to develop a predictive model for forecasting item sales across various outlets, which can help retailers optimize inventory, pricing strategies, and promotional activities. The dataset includes historical data on item sales, outlet details, and various item attributes. By leveraging this information, the model will predict future sales for each item at different outlets, allowing the business to anticipate demand trends, reduce stock-outs, and improve overall sales efficiency.


## Project description and objectives
- Sales Forecasting: Accurately predict the sales volume for individual items at multiple outlets.
- Insight Generation: Identify key drivers of sales, such as outlet characteristics, item attributes, seasonal trends, and pricing.on and Sources

## Business Problem
The business problem addressed in this project is accurately predicting item sales across different outlets. This prediction is essential for retailers to:


- Plan Strategically: Align marketing campaigns and resource allocation with expected demand trends.
- Analyze Performance: Identify underperforming products or stores for targeted improvements.
- Enhance Customer Satisfaction: Ensure desired products are available, boosting customer loyalty.
  Overall, effective sales predictions empower businesses to make informed decisions and improve profitability.

## Description of datasets used
The dataset for this sales prediction project consists of product and store-level information, capturing various attributes that contribute to sales performance across different retail outlets. Each row represents the sales of a specific item in a particular outlet. Here’s a breakdown of each attribute included in the dataset:

- Item_Identifier: A unique product ID representing each item in the dataset.
- Item_Weight: The weight of the product, which can affect sales, especially in categories where weight influences customer choice.
- Item_Fat_Content: An indicator of whether the product is low-fat or regular, capturing dietary attributes that may impact demand based on customer preferences.
- Item_Visibility: This metric reflects the percentage of the store’s total display area allocated to each product, offering insight into shelf placement and its influence on sales.
- Item_Type: The category to which each product belongs, such as beverages, snacks, or household items. Different categories may exhibit distinct sales trends.
- Item_MRP: The Maximum Retail Price of the product, indicating the list price. Price levels play a crucial role in customer purchasing decisions.
- Outlet_Identifier: A unique ID representing each store or outlet, allowing for sales tracking at the outlet level.
- Outlet_Establishment_Year: The year the store was established, which could correlate with factors like customer loyalty, brand recognition, or location stability.
- Outlet_Size: The size of the store (ground area covered), indicating store capacity and potentially influencing the volume of sales.
- Outlet_Location_Type: The type of area where the store is located, such as urban, suburban, or rural, as location impacts foot traffic and sales.
- Outlet_Type: Defines whether the outlet is a grocery store, supermarket type 1, 2, or 3. Different outlet types may attract varying customer demographics.
- Item_Outlet_Sales: The total sales of a product in a particular store. This is the target variable for prediction and is measured as the total sales volume, serving as the primary output of interest.
These features provide detailed information that enables the development of predictive models and analysis of sales patterns across items and outlets, helping to forecast future sales based on historical performance and store attributes.

## Data cleaning steps (handling missing values, outliers, etc.)
#### Data Preprocessing and Cleaning
In this project, we prepare data by handling missing values, encoding categorical variables, and scaling features to ensure the dataset is ready for accurate model training and testing. Below are the main steps:

#### Target and Feature Selection:

The target variable, Item_Outlet_Sales, represents the sales amount and is our primary focus for prediction.
We exclude Item_Identifier as it doesn’t provide predictive value.
#### Data Splitting:

The data is divided into training and testing sets to evaluate model performance effectively and avoid data leakage.
#### Identifying Column Types:

- Numerical Columns: Columns with numerical data types, which may include variables like Item_Weight and Item_MRP.
- Ordinal Columns: Columns with a clear, ordered relationship between categories, such as Item_Fat_Content, Outlet_Size, Outlet_Establishment_Year, and - - Outlet_Location_Type.
- Categorical Columns: The remaining object-type columns, excluding ordinal columns, representing different item and outlet characteristics.
#### Data Inspection:

For each ordinal column, we inspect value counts to understand the distribution of values and handle any missing or unusual data appropriately.
#### Pipeline Setup:

- Numerical Data: Missing values are imputed with the mean, and values are scaled for consistency across features.
- Categorical Data: Categories are encoded using one-hot encoding to represent each category as a unique binary vector.
- Ordinal Data: Missing values are imputed (using a placeholder if necessary), categories are ordinal-encoded based on specified orderings, and values are scaled.
#### Combining Pipelines:

We combine these transformations into a column transformer, allowing us to handle each data type in a single, consistent preprocessing pipeline.
#### Applying Transformations:

The column transformer is fitted to the training data and then applied to both the training and testing datasets, ensuring that all preprocessing steps are consistent and reproducible.
These preprocessing steps ensure that the data is fully prepared for use in predictive modeling, with encoded and scaled features ready for efficient training and evaluation. This setup also simplifies future processing, making it easy to apply transformations to new datasets.
#### Exploratory Data Analysis (EDA) and Summary statistics
![download](https://github.com/user-attachments/assets/39daf1e2-6c84-4661-88d2-a5ab3da220ea)

- The heat map shows only one positive strong relationship with a 0.57 correlation coeffeicient between the Item MRP and the Item Outlet Sales. In other words, when the MRP increase, the Item Outlet Sales increases too.
![download (1)](https://github.com/user-attachments/assets/0668402b-82e9-4bb7-a7f3-bcdbd4078e00)

- Outlet Type vs Outlet Item Sales:
The bar plot reveals that items sold in Supermarket Type1 have the highest total Item Outlet Sales, exceding 12 Million, followed by Supermarket Type3 at approximately 3M, and Supermarket Type2 at 2M. In contrast, Grocery Stores show a significantly lower total sales barely reaching 250 Thousand.
The strip plot illustrates the distribution of actual sales across each outlet type, with Supermarket Type 3 reaching a maximum value exceeding 13,000.
- Outlet Size vs Outlet Item Sales:
The bar plot shows that medium-sized outlets achieve the highest total Item Outlet Sales, exceeding 7.5M, followed by small outlets at around 4M, while large outlets have total sales around 2M.
- The strip plot illustrates the distribution of actual sales across each outlet size, with medium outlets reaching a maximum sales value exceeding 12,000.
Outlet Location Type vs Outlet Item Sales:
- The bar plot indicates that outlet located in Tier3 have the highest total Item Outlet Sales, around 7.5M, followed by Tier2 slightly exceding 6M. Then Tier1 outlets, with total sales around 4.5M.
- The strip plot shows the distribution of actual sales across each outlet location type, revealing that Tier3 has the highest actual sales reaching up to more than 12,000, the opposite of what might be assumed from the bar plot. This is followed by Tier2 and Tier1.

![download (2)](https://github.com/user-attachments/assets/f4119077-9aa9-482a-b477-cdecac5f2fcf)
- The item types that are achieving the highest total sales are:

Fruits and vegetables
Snack Foods
House Hold
Frozen Foods
The item types that are achieving the lowest total sales are:

Breakfast
Seafood
Others
![download (4)](https://github.com/user-attachments/assets/f57752aa-1fed-4e9c-afbf-2d0edee121c6)
- After segmenting the data according to the predefined price groups, as noticed in the very high-priced items, it shows a moderate negative correlation between the item visibility and the sales.
![download (5)](https://github.com/user-attachments/assets/42a446f4-7099-4afe-8069-fd4cbf92e3f6)
- Outlets in the dataset were established between 1999 and 2004. The data indicates that average sales for outlets established in most of these years are similar, with the exception of 1998. This discrepancy could be due to a lower volume of data for outlets built in 1998 compared to other years, as illustrated in the count plot. This limited data availability might skew the average sales figures for that year.


## Model Selection and Evaluation
- To determine the best model for predicting sales, we used both Linear Regression and Random Forest Regression models. Here’s an overview of each approach and its performance.

### - Linear Regression:

- Fitting: The model was trained using the transformed training data (X_train_tf).
- Model evaluation results: For training data, it achieved an R² of 0.562 with a Root Mean Squared Error (RMSE) of 1,139.104. On test data, it performed similarly, with an R² of 0.567 and an RMSE of 1,092.863. This indicates moderate predictive power and some room for improvement.

### Coefficients Plot
![feature_coefficients (3)](https://github.com/user-attachments/assets/2324eb45-57ba-4a53-94ed-418dec39a681)
### Interpretation of Coefficients
- Feature 1: Item MRP

  - Coefficient Value: 1000
  - Interpretation: "For each unit increase in the item's maximum retail price (MRP), the predicted outcome increases by 1000 units. This indicates that a higher MRP is associated with significantly higher sales or revenue."
- Feature 2: Outlet_Identifier_OUT027

   - Coefficient Value: -750 
   - Interpretation: "The coefficient for Outlet Identifier OUT027 is -750, which suggests that this particular outlet has lower sales compared to the baseline category. For each unit increase in this outlet's identifier, the predicted outcome decreases by 750 units."
- Feature 3: Outlet_Type_Supermarket Type1

   - Coefficient Value: 500 
   - Interpretation: "Being categorized as Supermarket Type1 results in an increase of 500 units in the predicted outcome. This means that this outlet type performs better in terms of sales compared to the reference category."
- Feature 4: Item_Type_Seafood

   - Coefficient Value: 300 
   - Interpretation: "For each unit increase in the sale of seafood items, the predicted outcome increases by 300 units. This suggests that seafood is a high-demand product in the sales context."
- Feature 5: Outlet_Location_Type

    - Coefficient Value: -200 
   - Interpretation: "The outlet's location type has a coefficient of -200, indicating that certain location types may be less favorable for sales, reducing the predicted outcome by 200 units."

### - Random Forest Regression:

- Model evaluation results: After training, Random Forest showed high accuracy on training data (R² = 0.938, RMSE = 426.955) but slightly lower performance on test data (R² = 0.559, RMSE = 1,103.635), suggesting potential overfitting.

### Feature Importances Plot
![feature_importances for random forest model](https://github.com/user-attachments/assets/a5a9b4a6-24a6-4707-922e-4d42b92dc2c5)
### Interpretation of Feature Importance
- Feature 1: Item MRP

   - Importance Value: 0.5 
    - Interpretation: "The item's maximum retail price (MRP) is the most significant predictor of sales in our model. An increase in the item's MRP is strongly associated with an increase in the predicted sales, indicating that higher-priced items are likely to generate more revenue."
- Feature 2: Outlet_Type_Grocery Store

   - Importance Value: 0.4 
    - Interpretation: "Being categorized as a Grocery Store significantly contributes to the model's predictions. This suggests that outlets identified as grocery stores perform better in terms of sales compared to other types, indicating a strong consumer preference for this outlet type."
- Feature 3: Outlet_Type_Supermarket Type3

  - Importance Value: 0.35 
   - Interpretation: "Supermarket Type3 is also a key feature influencing sales predictions. This type of outlet seems to attract a larger customer base, leading to higher sales volumes compared to the baseline outlet type."
- Feature 4: Outlet_Identifier_OUT027

   - Importance Value: 0.25 
    - Interpretation: "The specific outlet identifier OUT027 has a notable impact on sales, indicating that this particular outlet performs exceptionally well or poorly relative to others. This could provide insights into targeted marketing or operational improvements for this outlet."
- Feature 5: Outlet_Identifier_OUT010

   - Importance Value: 0.20
    - Interpretation: "This outlet identifier suggests a level of performance in terms of sales that is significant, albeit slightly less impactful than the top features. Understanding its customer base and sales strategies may be beneficial."


## Hyperparameter Tuning:
Using RandomizedSearchCV, we tuned hyperparameters, such as max_depth, n_estimators, min_samples_leaf, max_features, and oob_score. This - cross-validated approach identified the optimal configuration for improved performance.
Hyperparameter tuning (e.g., using GridSearchCV)


## Best Model: 
After tuning, the best model was retrained and evaluated, providing a refined model with enhanced generalizability for unseen data.
This selection process helped balance bias and variance, refining the model’s predictive accuracy for sales forecasting.


## Summary of model performance

- Summary of Model Performance
In our model comparison for sales prediction, we evaluated Tuned Random Forest, Default Random Forest, and Linear Regression based on their performance against the testing data:

- Tuned Random Forest vs. Default Random Forest:

The tuned Random Forest model exhibited a test R² value of 0.597, an improvement over the default Random Forest's R² of 0.559. This improvement indicates that the tuning process effectively enhanced model performance on the test data.
- Tuned Random Forest vs. Linear Regression:

The Linear Regression model achieved a test R² of 0.567, which is lower than the tuned Random Forest's R² of 0.597. This shows that the tuned Random Forest outperforms Linear Regression in predicting item sales.
- Recommendation:

The Tuned Random Forest model is recommended as the best model overall due to its superior test performance. It provides the best balance between training and testing data performance, despite some indications of overfitting.
- Model Insights:

The tuned Random Forest model explains approximately 60% of the variance in item sales, indicating that while it captures many factors influencing sales, around 40% remains unexplained, highlighting opportunities for further refinement.
The model's RMSE on the test data is approximately 1,054 dollars, suggesting that, on average, the model's predictions deviate from actual sales figures by about this amount. RMSE was chosen for its interpretability and ability to penalize larger prediction errors, making it a suitable metric for this analysis.
In summary, the tuned Random Forest model stands out as the most effective predictor of item sales, with a solid balance of accuracy and interpretability, although there is still potential for further enhancement.

### Final Recommendations for Stakeholders
- Based on the analysis of both the linear regression coefficients and the feature importances from the Random Forest model, here are some recommendations:

   - Optimize Pricing Strategy: Since the item's MRP has a strong correlation with sales, consider revising pricing strategies to enhance revenue, especially for high-demand items.

   - Evaluate Supermarket Types: Investigate the performance of Supermarket Type3 to understand what makes it successful, and replicate these practices in other outlet types.


  - Monitor Outlet Performance: Continuously monitor the performance of the identified outlets and implement strategies that leverage their strengths to enhance overall sales.

