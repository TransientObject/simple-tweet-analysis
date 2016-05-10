# simple-tweet-analysis
About collecting tweets 

AND 

Extracting basic data


Usage
----- 

Download [TIGER shapefiles](https://www.census.gov/geo/maps-data/data/tiger-line.html) into shapes/2015 folder

## Examples

    Count english tweets containing each of the keywords

        python analyze.py -s df -d tweets.json -o kw_df_count.json -kw keyword_terms.txt -en
