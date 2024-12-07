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

4. One step I took in this initial data analysis was to quantify the column 'review'. The actual written review does not provide us much information on its own, so I searched each review for positive words such as "good", "excellent", and "great". By counting positive words, I thought that we could check how positive the written review is. On average, we would expect 5 star reviews to have many positive words. We will check this idea later in the bivariate analysis section.

5. Finally, we can remove any columns that we will not use in our model or analysis. We will only need:

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