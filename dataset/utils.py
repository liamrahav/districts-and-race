"""
Utility functions for working with the raw datasets.
"""

def clean_underscores(x):
  if '__' in x:
    return ' '.join(x.split('__'))
  return x