---
title: "Pandas DataFrames Essential Patterns"
date: 2024-12-13
draft: false
category: "python"
tags: ["pandas", "dataframes", "data-analysis", "python"]
---

Essential patterns for working with Pandas DataFrames: creation, manipulation, filtering, aggregation, and transformation.

## Installation

```bash
pip install pandas numpy
```

## Creating DataFrames

### From Dictionary

```python
import pandas as pd
import numpy as np

# From dict of lists
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# From list of dicts
data = [
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Bob', 'age': 30, 'city': 'LA'},
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}
]
df = pd.DataFrame(data)

# From dict of Series
df = pd.DataFrame({
    'A': pd.Series([1, 2, 3]),
    'B': pd.Series([4, 5, 6])
})
```

### From Files

```python
# CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep=';', encoding='utf-8')
df = pd.read_csv('data.csv', parse_dates=['date_column'])

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM table', conn)

# Parquet
df = pd.read_parquet('data.parquet')

# From clipboard
df = pd.read_clipboard()
```

### Generate Sample Data

```python
# Random data
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randint(0, 100, 100),
    'C': np.random.choice(['X', 'Y', 'Z'], 100)
})

# Date range
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(100)
})

# From NumPy array
arr = np.random.randn(5, 3)
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

---

## Inspecting DataFrames

```python
# Basic info
df.head()           # First 5 rows
df.head(10)         # First 10 rows
df.tail()           # Last 5 rows
df.sample(5)        # Random 5 rows
df.shape            # (rows, columns)
df.columns          # Column names
df.dtypes           # Data types
df.info()           # Summary info
df.describe()       # Statistical summary
df.memory_usage()   # Memory usage

# Quick stats
df['column'].value_counts()
df['column'].nunique()
df['column'].unique()
df.isnull().sum()   # Count nulls per column
df.duplicated().sum()  # Count duplicates
```

---

## Selecting Data

### Column Selection

```python
# Single column (returns Series)
df['name']
df.name  # Attribute access (if no spaces in name)

# Multiple columns (returns DataFrame)
df[['name', 'age']]

# Select by dtype
df.select_dtypes(include=['int64', 'float64'])
df.select_dtypes(exclude=['object'])

# Column slicing
df.iloc[:, 0:3]  # First 3 columns
```

### Row Selection

```python
# By index
df.iloc[0]          # First row
df.iloc[0:5]        # First 5 rows
df.iloc[[0, 2, 4]]  # Specific rows

# By label
df.loc[0]           # Row with index 0
df.loc[0:5]         # Rows 0 through 5 (inclusive)
df.loc[[0, 2, 4]]   # Specific rows

# Boolean indexing
df[df['age'] > 30]
df[df['city'] == 'NYC']
df[(df['age'] > 25) & (df['city'] == 'NYC')]
df[df['age'].between(25, 35)]
df[df['name'].str.contains('Alice')]
df[df['name'].isin(['Alice', 'Bob'])]
df[~df['name'].isin(['Alice'])]  # NOT in list

# Query method
df.query('age > 30')
df.query('age > 30 and city == "NYC"')
df.query('name in ["Alice", "Bob"]')
```

### Combined Selection

```python
# Row and column
df.loc[0:5, ['name', 'age']]
df.iloc[0:5, 0:2]

# Boolean + columns
df.loc[df['age'] > 30, ['name', 'city']]
```

---

## Filtering Patterns

```python
# Multiple conditions
mask = (df['age'] > 25) & (df['city'].isin(['NYC', 'LA']))
filtered = df[mask]

# Filter by string patterns
df[df['name'].str.startswith('A')]
df[df['name'].str.endswith('e')]
df[df['name'].str.contains('li', case=False)]
df[df['email'].str.match(r'.*@gmail\.com')]

# Filter by null values
df[df['column'].isnull()]
df[df['column'].notnull()]
df.dropna()  # Drop rows with any null
df.dropna(subset=['column'])  # Drop if specific column is null
df.dropna(how='all')  # Drop only if all values are null

# Filter by index
df[df.index.isin([0, 2, 4])]
df.loc['2024-01-01':'2024-01-31']  # Date range

# Top/bottom N
df.nlargest(10, 'age')
df.nsmallest(10, 'age')

# Sample
df.sample(n=10)  # Random 10 rows
df.sample(frac=0.1)  # Random 10%
df.sample(n=10, random_state=42)  # Reproducible
```

---

## Adding/Modifying Columns

```python
# Simple assignment
df['new_col'] = 0
df['new_col'] = df['age'] * 2
df['full_name'] = df['first_name'] + ' ' + df['last_name']

# Conditional assignment
df['category'] = df['age'].apply(lambda x: 'young' if x < 30 else 'old')
df['category'] = np.where(df['age'] < 30, 'young', 'old')
df['category'] = pd.cut(df['age'], bins=[0, 30, 60, 100], 
                         labels=['young', 'middle', 'senior'])

# Multiple conditions
conditions = [
    df['age'] < 25,
    (df['age'] >= 25) & (df['age'] < 40),
    df['age'] >= 40
]
choices = ['young', 'middle', 'senior']
df['age_group'] = np.select(conditions, choices, default='unknown')

# Apply function
df['age_squared'] = df['age'].apply(lambda x: x**2)
df['name_length'] = df['name'].str.len()

# Map values
mapping = {'NYC': 'New York', 'LA': 'Los Angeles'}
df['city_full'] = df['city'].map(mapping)

# Replace values
df['city'] = df['city'].replace({'NYC': 'New York', 'LA': 'Los Angeles'})

# Insert at specific position
df.insert(1, 'new_col', 0)  # Insert at position 1

# Rename columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']  # Rename all
```

---

## Aggregation and Grouping

### GroupBy Operations

```python
# Basic groupby
grouped = df.groupby('city')
grouped.size()  # Count per group
grouped.mean()  # Mean of numeric columns
grouped.sum()
grouped.count()
grouped.min()
grouped.max()
grouped.std()

# Multiple columns
df.groupby(['city', 'age_group']).size()

# Specific aggregation
df.groupby('city')['age'].mean()
df.groupby('city')['age'].agg(['mean', 'min', 'max', 'std'])

# Multiple aggregations
df.groupby('city').agg({
    'age': ['mean', 'min', 'max'],
    'salary': ['sum', 'mean']
})

# Custom aggregation
df.groupby('city')['age'].agg(
    mean_age='mean',
    min_age='min',
    max_age='max',
    range_age=lambda x: x.max() - x.min()
)

# Apply custom function
def custom_func(group):
    return group['age'].max() - group['age'].min()

df.groupby('city').apply(custom_func)

# Transform (keeps original shape)
df['age_normalized'] = df.groupby('city')['age'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter groups
df.groupby('city').filter(lambda x: len(x) > 10)
df.groupby('city').filter(lambda x: x['age'].mean() > 30)
```

### Pivot Tables

```python
# Basic pivot
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='department',
    aggfunc='mean'
)

# Multiple aggregations
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='department',
    aggfunc=['mean', 'sum', 'count']
)

# Multiple values
pivot = df.pivot_table(
    values=['salary', 'age'],
    index='city',
    columns='department',
    aggfunc='mean'
)

# With margins (totals)
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='department',
    aggfunc='mean',
    margins=True
)

# Cross-tabulation
pd.crosstab(df['city'], df['department'])
pd.crosstab(df['city'], df['department'], normalize='all')  # Percentages
```

---

## Sorting

```python
# Sort by single column
df.sort_values('age')
df.sort_values('age', ascending=False)

# Sort by multiple columns
df.sort_values(['city', 'age'], ascending=[True, False])

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# In-place sorting
df.sort_values('age', inplace=True)
```

---

## Merging and Joining

```python
# Merge (SQL-like join)
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

# Inner join (default)
pd.merge(df1, df2, on='key')

# Left join
pd.merge(df1, df2, on='key', how='left')

# Right join
pd.merge(df1, df2, on='key', how='right')

# Outer join
pd.merge(df1, df2, on='key', how='outer')

# Multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])

# Different column names
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Merge on index
pd.merge(df1, df2, left_index=True, right_index=True)

# Concatenate
pd.concat([df1, df2])  # Vertical (rows)
pd.concat([df1, df2], axis=1)  # Horizontal (columns)
pd.concat([df1, df2], ignore_index=True)  # Reset index

# Append (deprecated, use concat)
pd.concat([df1, df2], ignore_index=True)

# Join (on index)
df1.join(df2, how='left')
```

---

## Reshaping

```python
# Melt (wide to long)
df_wide = pd.DataFrame({
    'id': [1, 2],
    'A': [10, 20],
    'B': [30, 40]
})
df_long = df_wide.melt(id_vars=['id'], var_name='variable', value_name='value')

# Pivot (long to wide)
df_wide = df_long.pivot(index='id', columns='variable', values='value')

# Stack/Unstack
df.stack()    # Columns to rows
df.unstack()  # Rows to columns

# Transpose
df.T
```

---

## Handling Missing Data

```python
# Detect missing
df.isnull()
df.notnull()
df.isnull().sum()  # Count per column

# Drop missing
df.dropna()  # Drop rows with any null
df.dropna(axis=1)  # Drop columns with any null
df.dropna(how='all')  # Drop only if all values null
df.dropna(subset=['column'])  # Drop if specific column null
df.dropna(thresh=2)  # Keep rows with at least 2 non-null values

# Fill missing
df.fillna(0)  # Fill with constant
df.fillna(method='ffill')  # Forward fill
df.fillna(method='bfill')  # Backward fill
df.fillna(df.mean())  # Fill with mean
df.fillna({'col1': 0, 'col2': 'unknown'})  # Different values per column

# Interpolate
df['column'].interpolate()
df['column'].interpolate(method='linear')
df['column'].interpolate(method='polynomial', order=2)

# Replace
df.replace(np.nan, 0)
df.replace([0, -999], np.nan)
```

---

## Time Series Operations

```python
# Create datetime index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample
df.resample('D').mean()  # Daily average
df.resample('W').sum()   # Weekly sum
df.resample('M').last()  # Monthly last value
df.resample('Q').agg({'value': 'sum', 'count': 'size'})

# Rolling window
df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['rolling_std'] = df['value'].rolling(window=7).std()
df['rolling_sum'] = df['value'].rolling(window=7).sum()

# Expanding window
df['expanding_mean'] = df['value'].expanding().mean()

# Shift
df['prev_value'] = df['value'].shift(1)  # Previous row
df['next_value'] = df['value'].shift(-1)  # Next row
df['pct_change'] = df['value'].pct_change()  # Percentage change

# Date operations
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter

# Filter by date
df['2024-01-01':'2024-01-31']
df.loc['2024-01']  # All of January 2024
```

---

## String Operations

```python
# String methods (vectorized)
df['name'].str.lower()
df['name'].str.upper()
df['name'].str.title()
df['name'].str.strip()
df['name'].str.replace('old', 'new')
df['name'].str.split(' ')
df['name'].str.split(' ', expand=True)  # Split into columns
df['name'].str.len()
df['name'].str.contains('pattern')
df['name'].str.startswith('A')
df['name'].str.endswith('e')
df['name'].str.extract(r'(\d+)')  # Regex extraction
df['name'].str.findall(r'\d+')
df['name'].str.count('a')
df['name'].str.slice(0, 3)  # First 3 characters
```

---

## Performance Optimization

```python
# Use categorical for repeated strings
df['category'] = df['category'].astype('category')

# Downcast numeric types
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')

# Use chunks for large files
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)

# Vectorized operations (avoid loops)
# ❌ Slow
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col1'] * 2

# ✅ Fast
df['new_col'] = df['col1'] * 2

# Use query for complex filters
df.query('age > 30 and city == "NYC"')  # Faster than boolean indexing

# Use eval for complex calculations
df.eval('new_col = col1 + col2 * col3')

# Copy vs view
df_copy = df.copy()  # Full copy
df_view = df[['col1', 'col2']]  # View (faster, but modifies original)
```

---

## Export Data

```python
# CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', sep=';', encoding='utf-8')

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# JSON
df.to_json('output.json', orient='records')

# SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)

# Parquet
df.to_parquet('output.parquet', compression='gzip')

# HTML
df.to_html('output.html')

# Clipboard
df.to_clipboard()
```

---

## Common Patterns

### Remove Duplicates

```python
# Drop duplicate rows
df.drop_duplicates()
df.drop_duplicates(subset=['column'])
df.drop_duplicates(keep='first')  # Keep first occurrence
df.drop_duplicates(keep='last')   # Keep last occurrence
df.drop_duplicates(keep=False)    # Remove all duplicates
```

### Reset Index

```python
df.reset_index(drop=True)  # Drop old index
df.reset_index()  # Keep old index as column
```

### Set Index

```python
df.set_index('column')
df.set_index(['col1', 'col2'])  # MultiIndex
```

### Chain Operations

```python
result = (df
    .query('age > 25')
    .groupby('city')
    .agg({'salary': 'mean', 'age': 'max'})
    .reset_index()
    .sort_values('salary', ascending=False)
    .head(10)
)
```

### Conditional Replacement

```python
# Replace values conditionally
df.loc[df['age'] > 30, 'category'] = 'senior'
df.loc[df['city'] == 'NYC', 'region'] = 'East'
```

---

## Further Reading

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Effective Pandas](https://github.com/TomAugspurger/effective-pandas)

