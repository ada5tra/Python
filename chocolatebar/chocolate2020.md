# Context


School project based on exploring a dataset about chocolate ratings, its ingredients and countries of origin, etc. EDA and development of machine learning algorithms to predict a target variable, and exploring some of pyspark functionalities.

- Predicting of chocolate bars ratings
- topic modelling on ingredients/flavour tastings.


# info


Chocolate is one of the most popular candies in the world. Each year, residents of the United States collectively eat more than 2.8 billion pounds. However, not all chocolate bars are created equal! This dataset contains expert ratings of over 1,700 individual chocolate bars, along with information on their regional origin, percentage of cocoa, the variety of chocolate bean used, and where the beans were grown.


## Rating Scale
Flavors of Cacao Rating System:

4.0 - 5.0 = Outstanding

3.5 - 3.9 = Highly Recommended

3.0 - 3.49 = Recommended

2.0 - 2.9 = Disappointing

1.0 - 1.9 = Unpleasant

*Not all the bars in each range are considered equal, so to show variance from bars in the same range I have assigned .25, .50 or .75.

Each chocolate is evaluated from a combination of both objective qualities and subjective interpretation. A rating here only represents an experience with one bar from one batch. Batch numbers, vintages, and review dates are included in the database when known. I would recommend people to try all the chocolate on the database regardless of the rating and experience for themselves.

The database is narrowly focused on plain dark chocolate to appreciate the flavors of the cacao when made into chocolate. The ratings do not reflect health benefits, social missions, or organic status.

The flavor is the most important component of the Flavors of Cacao ratings. Diversity, balance, intensity, and purity of flavors are all considered. A straight forward single note chocolate can rate as high as a complex flavor profile that changes throughout. Genetics, terroir, post-harvest techniques, processing, and storage can all be discussed when considering the flavor component.

Texture has a great impact on the overall experience and it is also possible for texture related issues to impact flavor. It is a good way to evaluate the makers' vision, attention to detail, and level of proficiency.

Aftermelt is the experience after the chocolate has melted. Higher quality chocolate will linger and be long-lasting and enjoyable. Since the after melt is the last impression you get from the chocolate, it receives equal importance in the overall rating.

Overall Opinion is really where the ratings reflect a subjective opinion. Ideally, it is my evaluation of whether or not the components above worked together and opinion on the flavor development, character, and style. It is also here where each chocolate can usually be summarized by the most prominent impressions that you would remember about each chocolate
Acknowledgements

These ratings were compiled by Brady Brelinski, Founding Member of the Manhattan Chocolate Society. For up-to-date information, as well as additional content (including interviews with craft chocolate makers), please see his website: Flavors of Cacao


**Inspiration**

We have multiple questions to answer, in the below list we answer most important pieces of information that possible to answer.\

    Where are the best cocoa beans grown?
    Which countries produce the highest-rated bars?
    Who creates the best Chocolate bars?
    What is Favorite taste?
    Which company has highest Rate?
    
    
Data source: (https://www.kaggle.com/soroushghaderi/chocolate-bar-2020)
