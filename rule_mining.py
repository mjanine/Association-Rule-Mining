import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Load dataset
data = pd.read_csv("Groceries_dataset.csv")
print("Sample records:")
print(data.head())

# Group items bought by each customer
transactions = data.groupby('Member_number')['itemDescription'].apply(list).values.tolist()

# Step 2: One-hot encode transactions
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
print("\nEncoded dataset dimensions:", df.shape)

# Step 3: Find frequent itemsets (support ≥ 0.01)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values('support', ascending=False, inplace=True)
print("\nFrequent itemsets preview:")
print(frequent_itemsets.head())

# Step 4: Generate and filter association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
filtered = rules[(rules['confidence'] > 0.6) & (rules['lift'] > 1.2)]
filtered = filtered.sort_values(by=['lift', 'confidence'], ascending=[False, False]).reset_index(drop=True)

# Step 5: Show and save top 5 rules
top_5 = filtered.head(5)
print("\nTop 5 Association Rules:")
print(top_5[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
top_5.to_csv("top_rules.csv", index=False)
print("\nTop 5 rules saved to 'top_rules.csv'.")

# Summary
print("\nSummary:")
print(f"Transactions analyzed: {len(transactions)}")
print(f"Frequent itemsets found: {len(frequent_itemsets)}")
print(f"Strong rules extracted: {len(filtered)}")
