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
- We begin by dropping unwanted values
  - unnamed: 0, blanl, blank2 were removed because they were empty/meaningless values
  - GS, Player were dropped because we didn't think GS would be a good predictor and Player name wouldn't also matter at all
  - ORB and DRB were removed because Total rebounds was a cumulative stat of the two
  - 2 pointers,3 pointers and FG made were all removed because they directly indicate points made and we believed it could cause problems with overfitting
- Next we filled in the null values of our columns
  - a
