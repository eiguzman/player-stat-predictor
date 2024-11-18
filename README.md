# CSE151AGroupProject
### How will you preprocess your data:
- to (one-hot) encode categorical data like Team, Positions
- for null/nan values, we will replace it with an adjusted mean values
- for each of the season total categories (PTS, TRB, AST), we will divide it by the games played to get the per game average.
- use min-max scaling to normalize the features


### How we preprocessed the data
#### Encoding
- We performed one-hot encoding for categorical variables Team (`Tm`) and Position(`Pos`).
- Included a `dummy_na=True` parameter to handle missing values during encoding.

#### Imputing Null Values
- We began by dropping our null values for Team, Player, Position, Year, and Games played because the percentage makeup of our data was very small
