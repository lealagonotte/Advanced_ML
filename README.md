# Multi arm bandit for movie recommender systems

## Presentation

This project implements and compares different multi arm bandits algorithms for movie recommendation. For more information about multi-arm bandit, consult the article _Introduction to Multi-Armed Bandits_.[\[3\]](#ref-intro)

The algorithms compared are :
- $\epsilon$-greedy
-  Multi-armed nearest-neighbor bandit [\[4\]](#ref-knn)
- Dynamic clustering based contextual combinatorial multi-armed bandit (DC3MAB) [\[2\]](#ref-DC3MAB)

## Data

We use a combination of synthetic data and real data. 

The real data commes from the MovieLens 1M data [\[1\]](#ref-movielens), widely adopted benchmark in recommendation system research, provided by the GroupLens Research Lab. The dataset contains anonymized
user-movie interactions in the form of explicit ratings, along with basic movie metadata such as titles
and genres. MovieLens is characterized by a high level of sparsity and strong heterogeneity in user
preferences, reflecting realistic recommendation scenarios.

## Running the code

To run the code, you need to install the libraries in requirements.txt:
```bash
pip install -r requirements.txt
```

To prepare the MovieLens data, run the notebook `data_preparation.ipynb` in the `data` folder:
```bash
jupyter nbconvert --to notebook --execute data.data_preparation.ipynb --inplace
```
This will create a `df_merged.csv` file.

To generate synthetic data, run the file `synthetic_data.py` in the `data` folder:
```bash 
python -m data.synthetic_data
```
To run $\epsilon$-greedy algorithms on synthetic data, run `synthetic_epsgreedy.ipynb`:
```bash
jupyter nbconvert --to notebook --execute synthetic_epsgreedy.ipynb --inplace
```

To run $\epsilon$-greedy algorithms on real data, run `epsgreedy.ipynb`:
```bash
jupyter nbconvert --to notebook --execute epsgreedy.ipynb --inplace
```

To run Multi-armed nearest-neighbor bandit algorithms, run `knn.ipynb`:
```bash
jupyter nbconvert --to notebook --execute knn.ipynb --inplace
```

To run DC3MAB algorithms, run `dynamic_clustering.ipynb`:
```bash
jupyter nbconvert --to notebook --execute dynamic_clustering.ipynb --inplace
```

## References

<a id="ref-movielens"></a> **[1]** Harper, F. Maxwell and Joseph A. Konstan (Dec. 2015). “The MovieLens Datasets: History and
Context”. In: ACM Transactions on Interactive Intelligent Systems (TIIS) 5.4, 19:1–19:19. DOI:
10.1145/2827872. URL: https://doi.org/10.1145/2827872.

<a id="ref-DC3MAB"></a> **[2]** Sanz-Cruzado, Javier, Pablo Castells, and Esther López (2019). “A simple multi-armed nearest-
neighbor bandit for interactive recommendation”. In: Proceedings of the 13th ACM Conference on
Recommender Systems. RecSys ’19. Copenhagen, Denmark: Association for Computing Machinery,
pp. 358–362. ISBN: 9781450362436. DOI: 10.1145/3298689.3347040. URL: https://doi.
org/10.1145/3298689.3347040.

<a id="ref-intro"></a> **[3]** Slivkins, Aleksandrs (2019). “Introduction to Multi-Armed Bandits”. In: Foundations and Trends in
Machine Learning 12.1–2, pp. 1–286.

<a id="ref-knn"></a> **[4]** Yan, Cairong et al. (2022). “Dynamic clustering based contextual combinatorial multi-armed ban-
dit for online recommendation”. In: Knowledge-Based Systems 257, p. 109927. ISSN: 0950-7051. DOI: https://doi.org/10.1016/j.knosys.2022.109927. URL: https://www.
sciencedirect.com/science/article/pii/S0950705122010206.