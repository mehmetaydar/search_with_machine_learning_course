import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
df['query'].str.lower()


# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
indexed_parents = parents_df.set_index('category')
current_counts = pd.DataFrame(df['category'].value_counts()).reset_index()
current_counts.columns = ['category', 'count']
current_counts = current_counts[current_counts['category'] != root_category_id]
min_count = current_counts['count'].min()

while(min_count < min_queries and len(df) > 0): 
    cats_to_replace = current_counts.loc[current_counts['count'] == min_count]
    replacements = {}
    for to_replace in set(cats_to_replace['category'].values):
        print(f'Replacing {to_replace}, n = {cats_to_replace.loc[cats_to_replace["category"] == to_replace, "count"].squeeze()}')
        replacements[to_replace] = indexed_parents.loc[to_replace]['parent']
        #df.loc[df['category'] == to_replace, 'category'] = indexed_parents.loc[to_replace]['parent']
    df['category'].replace(to_replace=replacements, regex=False, inplace=True)

    current_counts = pd.DataFrame(df['category'].value_counts()).reset_index()
    current_counts.columns = ['category', 'count']
    current_counts = current_counts[current_counts['category'] != root_category_id]
    min_count = current_counts['count'].min()

# Create labels in fastText format.
df['label'] = '__label__' + df['category']
df['output'] = df['label'] + '' +df['query']
df[['output']].to_csv(output_file_name, header=False, sep='I', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
