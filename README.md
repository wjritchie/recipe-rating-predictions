# recipe-rating-predictions
Portfolio Homework - University of Michigan EECS 398-003 (Practical Data Science)

# The Dataset

Our dataset, Recipes and Ratings, consists of two CSV files.

1. RAW_interactions, which consists of 731,928 rows. This dataset is made up of reviews of recipes, including the text of the review and a score ranging from 0 to 5.

2. RAW_recipes, which consists of 83,783 rows. This dataset is made up of recipes, including the ingredients they contain, the steps they contain, and how many minutes they take.

We will combine these two datasets using the recipe's ID, a shared column between the two CSV files. Using this, we'll look at the recipe ratings to see if they can be predicted using other features of the dataset, such as the recipe's complexity. The complexity in question is determined by the following columns:

- n_ingredients, the number of ingredients the recipe uses
- n_steps, the number of steps the recipe contains
- minutes, the time it is supposed to take to make the recipe


# Initial Data Cleaning and Analysis

To properly analyze this dataset, we must first clean it. Here are the detailed steps that were taken to clean up the data.

1. We first merge the two CSV files described above, RAW_interactions and RAW_recipes. RAW_interactions contains a column called 'recipe_id', and RAW_recipes contains a column called 'id'. Using these, we can merge the two datasets to create one even larger dataset, which we will call recipes.

2. Now that the datasets are combined, we have both 'recipe_id' and 'id' as columns within recipes. Therefore, we can remove 'id' because they are the same.

3. Another step we take here is filling in empty cells that have the value NaN. By checking which columns have NaN values (recipes.isna().sum()), we see that only the columns 'review', 'name' and 'description' contain empty values. Therefore, we can fill these in. Because each column is a string and we will not be using them, we can simply fill them with the default value of "No review", "No name" or "No description".

4. We have two columns that contain dates: 'date' and 'submitted'. We convert these to a datetime object using pandas.

5. One step I took in this initial data analysis was to quantify the column 'review'. The actual written review does not provide us much information on its own, so I searched each review for positive words such as "good", "excellent", and "great". By counting positive words, I thought that we could check how positive the written review is. On average, we would expect 5 star reviews to have many positive words. We will check this idea later in the bivariate analysis section.

6. If we look at the "minutes" column, we see many outliers, with several being greater than 180 days in length, and 2 different rows being as long as 2 years. These do not many sense to include, as almost the entire dataset falls within 1 day. So to avoid these outliers skewing the data, we will only focus on recipes that take a day or less to make by running : recipes = recipes[recipes['minutes'] <= 1440].

7. Finally, we can remove any columns that we will not use in our model or analysis. We will only need:

- rating
- n_steps
- n_ingredients
- positive
- minutes

So we run recipes = recipes[["rating","n_steps","n_ingredients","positive","minutes"]] to drop our unneeded columns.


```py
print(recipes.head().to_markdown(index=False))
```

|   rating |   n_steps |   n_ingredients |   positive |   minutes |
|---------:|----------:|----------------:|-----------:|----------:|
|        5 |         4 |               8 |          1 |        40 |
|        5 |         9 |              10 |          1 |        30 |
|        5 |        14 |              14 |          1 |        22 |
|        5 |        14 |              14 |          1 |        22 |
|        4 |         7 |              12 |          2 |        40 |


# Univariate Analysis : Ratings

In the plotly histogram below, we see the distribution of ratings for the recipes. From this histogram, we see that the rating are significantly skewed, the large majority of ratings are 5.

<iframe
  src="assets/ratings-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

# Bivariate Analysis : Positive Words Per Rating

In this plotly histogram, we look at the average number of positive words used for each possible value of "rating". This plot shows what we would expect- that more positive words are used in higher rated reviews. 

<iframe
  src="assets/positive-words-per-rating.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>

# Pivot Table : Complexity

In this plotly heatmap, a pivot table is shown. On the x axis is the number of ingredients used in a recipe, and the y axis is the number of steps contained in the recipe. The value for each box is the average number of minutes it takes to make the recipe. In this heatmap we see that generally, as a recipe has more steps and ingredients, it takes more time to complete.

<iframe
  src="assets/complexity-heatmap.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>

# Imputation : Ratings of Zero

In our data cleaning step, we filled in all empty values. So there are no NaN values to fill anymore. 

However, in our dataset, there are ratings of 0, 1, 2, 3, 4, and 5. Looking closer at ratings of 0, we see that they are reviews with no score, and therefore should be reclassified. For example, if someone writes a positive review but doesn't leave a rating, it would be included as a 0. This will lower our model accuracy, and we should reassign values of 0.

Mean imputation and median imputation would not necessarily work here, as they would simply change all ratings of 0 to ratings of 5 (because of how skewed our data is). So instead I used a predictive model to predict what each rating would be. This way, the distribution of ratings remains more similar to what it was before.

Here are the value counts for rating before imputation:

|   rating |   count |
|---------:|--------:|
|        5 |  168575 |
|        4 |   37142 |
|        0 |   14835 |
|        3 |    7137 |
|        1 |    2847 |
|        2 |    2344 |

And here are the value counts for rating after imputation:

|   rating |   count |
|---------:|--------:|
|        5 |  180955 |
|        4 |   39450 |
|        3 |    7258 |
|        1 |    2851 |
|        2 |    2366 |

# Our Prediction Problem

In this model, we will look to predict a recipe's rating. To do this, we will use the recipe's number of ingredients, number of steps, and minutes needed to make. Through this prediction, we hope to see whether a recipe's rating is influenced by and can be predicted by its complexity.

This is a regression problem, as we will predict the rating on a scale from 1 to 5. We will use the following columns for this prediction:

- n_ingredients
- n_steps
- minutes

We will use MSE to check how accurate our model is because we are using a regression model. This way, we can balance how far off our prediction is, rather than whether it is incorrect or not like using accuracy in a classification problem.

# Baseline Model








