# CSE151AGroupProject
### How will you preprocess your data:
- to (one-hot) encode categorical data like Team, Positions
- for null/nan values, we will replace it with an adjusted mean values
- for each of the season total categories (PTS, TRB, AST), we will divide it by the games played to get the per game average.
- use min-max to scale since we do
