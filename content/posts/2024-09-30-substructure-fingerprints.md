---
title: Chemical Search in Data Warehouses
description: Learn how to create custom chemical fingerprints using RDKit for efficient molecule filtering in data warehouses.
date: 2024-09-30
draft: false
tags: [chemistry, data engineering, python]
params:
  math: true
---

In the pharmaceutical industry, data is the foundation of innovation, enabling researchers to accelerate drug discovery, optimize synthesis processes, and predict biological activities. Among the various types of data, chemical information holds a critical role. However, handling large volumes of chemical data presents unique challenges, including scalability, searchability, and standardization. This is where data warehousing comes into play.

Data warehousing is the practice of integrating and organizing data from diverse sources into a centralized, accessible repository. In the context of the pharmaceutical industry, data warehousing for chemical information means consolidating the structures, properties, and experimental results related to chemical compounds into a single, well-structured platform. By providing a unified view, it becomes easier to query, analyze, and retrieve data, which is crucial for tasks such as lead optimization, structure-activity relationship analysis, and intellectual property management.

Effective data warehousing provides pharmaceutical researchers with fast and scalable access to structured chemical information, thereby enhancing the searchability of chemical compounds. Chemical searches often involve more than just exact match queries; they require substructure searches, similarity searches, and custom filtering based on molecular properties. To meet these needs, chemical fingerprints—bitstrings that encode structural information of a molecule—are employed as key indexing tools in data warehouses.

In this post, we’ll explore how customized chemical fingerprints, created using RDKit, can significantly improve the efficiency of chemical search processes within data warehouses. By tailoring fingerprint generation to the specific requirements of a project, researchers can optimize search results, enhance computational efficiency, and ultimately make more informed decisions in drug discovery.

## RDKit and molecular fingerprints

RDKit is an open-source toolkit for cheminformatics, widely used in the pharmaceutical industry for tasks like molecular modeling, property prediction, and chemical information retrieval. One of RDKit’s core features is the generation of molecular fingerprints—compact representations of molecular structures in the form of bitstrings. These fingerprints are used to encode the presence or absence of various chemical features or substructures, enabling efficient comparison of molecules.

In a typical fingerprinting process, RDKit analyzes the structure of a molecule and maps it to a binary vector, where each bit represents a specific substructure or chemical property. These fingerprints can be tailored for different types of chemical searches, such as similarity searching, which identifies molecules with related features, or substructure searching, which finds compounds that contain a particular fragment.

While RDKit offers a variety of pre-built fingerprint algorithms, such as Morgan, MACCS, and Atom Pair, customizing these fingerprints can offer significant advantages. By defining your own library of substructures, you can create a simple text-based fingerprint that highlights the presence of user-defined chemical patterns, allowing you to filter molecules based on specific features relevant to your research.

## Creating a custom `FilterCatalog` using SMARTS patterns

To create a custom filtering system for molecules using RDKit's `FilterCatalog`, you can define your own set of chemical substructures using SMARTS patterns and store them in a JSON file. This allows for easy customization and scalability, as you can modify the library of SMARTS patterns to suit specific project needs without hardcoding them.

In this example, we'll create a `FilterCatalog` from a JSON file that contains a dictionary where each key is a descriptive name for a chemical pattern, and each value is a corresponding SMARTS string.

### Prepare the JSON file with SMARTS Patterns

First, create a JSON file (`chemical_patterns.json`) with the following structure:

```json
{
  "AlkeneNonTerminal": "[C;D2;$(C([*;!$(*=*)])=C[*;!$(*=*)])]",
  "Amide": "[N;$([N;D2]([#6])C(=O)[#6]),$([N;D3]([#6])([#6])C(=O)[#6]),$([N;D1]C(=O)[#6])]",
  "Amine_Primary_Unsaturated_Aliphatic": "[N;X3;!+;!-;!$(N=*);$([N;!$(N~[*;!#6])]);!$(N[*]=[*;O,N]);D1;$([$(*[C]=[C]),$(*[C]#[C]),$(*[C]#[N]);!$(*a)])]",
  "Chloride_Aliphatic": "[Cl;$([Cl][*;$([#6]);!$(*=*)]);$([!$(*[C]=[C]);!$(*[C]#[C]);!$(*[C]#[N]);!$(*a)])]",
  "PrimaryAlcohol_Aliphatic": "[O;H1;D1;$(O[C;D2,D1;!$(C[a])]);!$(OC=*);!$(OC#*)]"
}
```

In this file:

- each key (e.g., "AlkeneNonTerminal") is a human-readable label for the pattern.
- each value is a SMARTS string that represents the chemical substructure.

### Read the JSON file and create a `FilterCatalog`

Now, let's create a custom `FilterCatalog` using RDKit. We'll read the JSON file and add each SMARTS pattern as a filter to the catalog.

Here's how you can do this for the following example:

![Molecule](https://raw.githubusercontent.com/f3rmion/f3rmion.github.io/refs/heads/main/_posts/assets/2024-09-30/molecule.png)

```python
import json

from rdkit.Chem import FilterCatalog
from rdkit import Chem

# Load SMARTS patterns from the JSON file
with open('chemical_patterns.json', 'r') as file:
    smarts_library = json.load(file)

# Create a custom FilterCatalog
catalog = FilterCatalog.FilterCatalog()

# Add each SMARTS pattern to the catalog
for pattern_name, smarts in smarts_library.items():
    # Create an RDKit molecule from the SMARTS pattern
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError("Invalid SMARTS pattern")

    # Create SMARTS matcher
    sm = FilterCatalog.SmartsMatcher(mol)

    # Create a FilterCatalogEntry for each SMARTS pattern
    entry = FilterCatalog.FilterCatalogEntry(pattern_name, sm)

    # Add the entry to the FilterCatalog
    catalog.AddEntry(entry)

# Test with an example molecule
mol = Chem.MolFromSmiles("C1C=CC=C(/C(/C(=O)NCCCl)=C\OC)C=1")

# Extract keys for every match and create a sorted substructure fingerprint
matches = catalog.GetMatches(mol)
matches_keys = [match.GetDescription() for match in matches]
matches_keys.sort()

# concatenate the substructure fingerprint
substructure_fingerprint = " ".join(matches_keys)

print(substructure_fingerprint)
# SMILES: C1C=CC=C(/C(/C(=O)NCCCl)=C\OC)C=1
# Substructure fingerprint: AlkeneNonTerminal Amide Chloride_Aliphatic
```

Explanation:

A. Loading SMARTS Patterns: The `json.load()` function is used to read the JSON file.

B. Creating the FilterCatalog:

- An empty `FilterCatalog` object (catalog) is instantiated.
- For each entry in the _smarts_library_, a SMART matcher object is created and added together with its _pattern_name_ to a `FilterCatalogEntry`.
- The entry is then added to the `FilterCatalog` via the `AddEntry` method of the catalog.

C. Testing a Molecule:

- A sample molecule is created from a SMILES string using `Chem.MolFromSmiles()`.
- The `catalog.GetMatches(mol)` method checks if the molecule contains patterns defined in the custom filter catalog.
- Keys of the matches are extracted, sorted, and concatenated to obtain the final substructure fingerprint.

Benefits of This Approach:

- Customizability: You can update the JSON file with new SMARTS patterns without modifying the core code. However, a new versions of the SMARTS library will result in updates in the data warehouse to keep your records up-to-date.
- Scalability: This approach scales well, allowing you to build extensive catalogs of chemical filters specific to the needs of different projects (FYI: I tested it for library sizes up to about 500 SMARTS patterns).
- Readability: Each SMARTS pattern is given a descriptive label, making it easy to understand and manage the chemical filters.

## Serializing the `FilterCatalog`

After creating a custom `FilterCatalog`, it can be beneficial to serialize it for future use without having to recreate it from scratch each time. RDKit provides functionality for serializing and saving filter catalogs, allowing you to store them as binary files. This can be particularly useful when working with large catalogs or when needing to share filters across different projects. You can use Python's pickle module to serialize the FilterCatalog object:

```python
import json
import os
import pickle

from rdkit import Chem
from rdkit.Chem import FilterCatalog

# Load SMARTS patterns from the JSON file
with open('chemical_patterns.json', 'r') as file:
    smarts_library = json.load(file)

# Create a custom FilterCatalog
catalog = FilterCatalog.FilterCatalog()

# Add each SMARTS pattern to the catalog
for pattern_name, smarts in smarts_library.items():
    # Create an RDKit molecule from the SMARTS pattern
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError("Invalid SMARTS pattern")

    # Create SMARTS matcher
    sm = FilterCatalog.SmartsMatcher(mol)

    # Create a FilterCatalogEntry for each SMARTS pattern
    entry = FilterCatalog.FilterCatalogEntry(pattern_name, sm)

    # Add the entry to the FilterCatalog
    catalog.AddEntry(entry)

# Serialize FilterCatalog
data = pickle.dumps(catalog.Serialize())
with open(os.path.join(os.getcwd(), "filter_catalog.pkl"), "wb") as write_file:
    write_file.write(data)
```

## Using LRU cache to deserialize the `FilterCatalog`

To manage the deserialization of a `FilterCatalog` efficiently, you can use Python's `functools.lru_cache` (`maxsize=1`). This allows you to keep the deserialized catalog in memory for rapid access, while automatically managing the cache size based on usage.

```python
import pickle

from functools import lru_cache

from rdkit.Chem import FilterCatalog

@lru_cache(maxsize=1)
def load_filter_catalog(binary_name: str = None) -> FilterCatalog.FilterCatalog:
    """Load RDKit SMARTS filter catalog for substructure search.

    Returns:
    RDKit FilterCatalog for SMARTS substructure search.
    """
    if not binary_name:
        raise FileNotFoundError("Binary name is required to load the filter catalog.")

    # Load the serialized filter catalog in file context
    with open(binary_name, "rb") as binary:
        catalog = pickle.load(binary)

    # deserialize the filter catalog
    return FilterCatalog.FilterCatalog(catalog)

# Load the filter catalog using the helper function
filter_catalog = load_filter_catalog("filter_catalog.pkl")
```

Explanation:

A. LRU Cache Decorator:

- The `@lru_cache(maxsize=1)` decorator caches the `load_filter_catalog()` function, ensuring that the `FilterCatalog` is loaded from disk only once, after which it stays in memory.
- The `maxsize=1` parameter ensures that only the most recently used catalog is kept in memory, making this solution memory-efficient while still providing fast access.

B. Loading the FilterCatalog:

- The function `load_filter_catalog` reads the serialized `FilterCatalog` from a binary file (_filter_catalog.pkl_) using pickle.
- Each time the function is called, if the cached version is already available, it returns it without reloading from the file.

Benefits of Using LRU Cache:

- Efficient Access: Deserializing from disk can be time-consuming. Caching the `FilterCatalog` ensures efficient access without the repeated I/O operations.
- Memory Management: By limiting the cache size with `maxsize=1`, you avoid overloading memory, especially if dealing with multiple objects or a large dataset.

## Leveraging custom fingerprints in data warehouses

In this blog post, we explored how RDKit can be used to create customized chemical fingerprints for efficient molecule filtering. By building a `FilterCatalog` with custom SMARTS patterns, we can tailor the fingerprinting process to fit specific chemical features of interest. This is particularly valuable when we need to focus on specific substructures or chemical groups in a large dataset.

We also saw how these custom fingerprints can be serialized for future use and cached in memory using an LRU cache, providing both scalability and efficiency in managing chemical data.

One of the key advantages of creating simple, text-based fingerprints is their compatibility with data warehouses. In systems, where text-based queries are extremely efficient, storing fingerprints as text makes it straightforward to filter out or select molecules based on predefined criteria. This allows researchers to leverage the speed of exact text searches for scalable chemical screening.

Using custom fingerprints in this way enables rapid retrieval of relevant molecules without complex and computationally costly substructure searches, making it an ideal solution for integrating cheminformatics into big data platforms. This approach ensures that the data warehouse maintains the performance required for large-scale chemical screening while providing the flexibility to refine and update chemical filters according to the evolving needs of a research project.
