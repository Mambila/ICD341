#!/usr/bin/env python
# coding: utf-8

# In[2]:


""" # Maize Production Analysis

## Introduction
This project analyses maize production using Python statistical computing tools.

## Objectives
1. Calculate average production.
2. Measure production variability.
3. Visualize maize production.
4. Interpret the results. """


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


np.random.seed(123)
production = np.random.normal(loc=5, scale=1.2, size=100)

df = pd.DataFrame({"Maize Production": production})
df.to_csv("../data/maize_data.csv", index=False)

print(df.head())


# In[5]:


print("Mean:", np.mean(production))
print("Standard deviation:", np.std(production))
print("Maximum:", np.max(production))
print("\nSummary Statistics:\n", df.describe())


# In[6]:


plt.hist(df["Maize Production"], bins=15, edgecolor="black")
plt.title("Maize Production Distribution")
plt.xlabel("Production (tonnes)")
plt.ylabel("Frequency")
plt.savefig("../outputs/maize_histogram.png", dpi=300, bbox_inches="tight")
plt.show()


# In[10]:


""" Interpretation
The analysis provides information about average maize production (~5 tonnes) and its variability. The histogram shows the distribution of the generated production observations.

## Conclusion
Python statistical computing tools can be used to generate, analyse and visualize agricultural production data. """


# In[ ]:




