import pandas as pd
df = pd.read_csv(
    'dataset/SMSSpamCollection',
    sep='\t',
    header=None,
    names=['label','message'],
    encoding= 'utf-8'
)

# Remove missing messages
df= df.dropna(subset=['message'])
# Remove duplicate messages 
df= df.drop_duplicates(subset=['message'])
# Normalize labels in same format
df['label'] = df['label'].str.lower().str.strip()
# Claen whitespaces
#df['message']= df['message'].str.replace(r'\s+','', regex=True)
df['message']= df['message'].str.strip()

# Save as a new clean csv dataset
df.to_csv('dataset/clean-dataset2.csv', index=False)
print('Cleaned Dataset Created Successfully.')
