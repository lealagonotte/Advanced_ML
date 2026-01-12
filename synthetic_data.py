import pandas as pd 
import numpy as np
from scipy.special import expit
from sklearn.cluster import KMeans

# Generate synthetic data with latent factors

def simulate_rating_matrix(
    n_users=1000,
    n_movies=5_000,
    n_factors=10,
    user_factor_std=1.0,
    movie_factor_std=10.0,
    random_state=12
):
    rng = np.random.default_rng(random_state)

    # Generate latent factors
    U = rng.normal(0, user_factor_std, size=(n_users, n_factors))
    V = rng.normal(0, movie_factor_std, size=(n_movies, n_factors))

    #  Compute affinity scores (Latent preference)
    scores = U @ V.T 
    
    #  Transform scores to probabilities [0, 1] using sigmoid
    # expit is equivalent to 1 / (1 + exp(-x))
    probs = expit(scores)
    
    # Generate Noisy Ratings using Shifted Binomial Distribution
    # We sample from B(n=4, p) -> gives integers in [0, 4]
    # Then add 1 -> gives integers in [1, 5]
    # This adds stochasticity: a user with high probability p=0.9 can still rate 4 instead of 5.
    raw_ratings = rng.binomial(n=4, p=probs)
    ratings = 1 + raw_ratings
    
    # Create Binary Ratings (Standard approach: >= 4 is positive)
    binary_ratings = (ratings >= 4).astype(int)

    
    user_factors_df = pd.DataFrame(U, columns=[f"factor_{k}" for k in range(n_factors)])
    user_factors_df.insert(0, "user_id", np.arange(n_users))
    movie_factors_df = pd.DataFrame(V, columns=[f"factor_{k}" for k in range(n_factors)])
    movie_factors_df.insert(0, "movie_id", np.arange(n_movies))

    user_ids = np.repeat(np.arange(n_users), n_movies)
    movie_ids = np.tile(np.arange(n_movies), n_users)

    interactions_df = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings.ravel(),
        "binary_rating": binary_ratings.ravel(), # Fixed typo "binary_ratings" -> "binary_rating"
        "true_prob": probs.ravel() # Optional: useful to debug or compute "Regret" vs ideal proba
    })

    return ratings, user_factors_df, movie_factors_df, interactions_df

ratings, user_factors_df, movie_factors_df, interactions_df = simulate_rating_matrix()

synthetic_ratings = interactions_df.rename(columns={
    'user_id': 'UserID',
    'movie_id': 'MovieID'   
})

synthetic_ratings.to_csv('synthetic_ratings.csv')



# Generate synthetic data with latent factors and communauties

def simulate_rating_matrix_communauties(
    n_users=1000,
    n_movies=5_000,
    n_factors=10,
    n_communities = 5,
    community_std = 0.3,       # variance intra-communauté (petite)
    community_mean_std = 2.0,  # écart entre communautés (grand)
    movie_factor_std=10.0,
    random_state=12
):
    rng = np.random.default_rng(random_state)

    # Generate latent factors
    # Assigner chaque utilisateur à une communauté
    user_communities = rng.integers(0, n_communities, size=n_users)

    # Moyennes latentes par communauté
    community_means = rng.normal(
        0,
        community_mean_std,
        size=(n_communities, n_factors)
    )

    # Génération des facteurs utilisateurs
    U = np.zeros((n_users, n_factors))

    for c in range(n_communities):
        idx = user_communities == c
        U[idx] = rng.normal(
            loc=community_means[c],
            scale=community_std,
            size=(idx.sum(), n_factors)
        )

    V = rng.normal(0, movie_factor_std, size=(n_movies, n_factors))

    #  Compute affinity scores (Latent preference)
    scores = U @ V.T 
    
    #  Transform scores to probabilities [0, 1] using sigmoid
    # expit is equivalent to 1 / (1 + exp(-x))
    probs = expit(scores)
    
    # Generate Noisy Ratings using Shifted Binomial Distribution
    # We sample from B(n=4, p) -> gives integers in [0, 4]
    # Then add 1 -> gives integers in [1, 5]
    # This adds stochasticity: a user with high probability p=0.9 can still rate 4 instead of 5.
    raw_ratings = rng.binomial(n=4, p=probs)
    ratings = 1 + raw_ratings
    
    # Create Binary Ratings (Standard approach: >= 4 is positive)
    binary_ratings = (ratings >= 4).astype(int)

    
    user_factors_df = pd.DataFrame(U, columns=[f"factor_{k}" for k in range(n_factors)])
    user_factors_df.insert(0, "user_id", np.arange(n_users))
    user_factors_df["community_id"] = user_communities
    movie_factors_df = pd.DataFrame(V, columns=[f"factor_{k}" for k in range(n_factors)])
    movie_factors_df.insert(0, "movie_id", np.arange(n_movies))

    user_ids = np.repeat(np.arange(n_users), n_movies)
    movie_ids = np.tile(np.arange(n_movies), n_users)

    interactions_df = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings.ravel(),
        "binary_rating": binary_ratings.ravel(), # Fixed typo "binary_ratings" -> "binary_rating"
        "true_prob": probs.ravel() # Optional: useful to debug or compute "Regret" vs ideal proba
    })

    return ratings, user_factors_df, movie_factors_df, interactions_df

ratings_comm, user_factors_df_comm, movie_factors_df_comm, interactions_df_comm = simulate_rating_matrix_communauties()
synthetic_ratings_comm = interactions_df_comm.merge(
    user_factors_df_comm[["user_id", "community_id"]],
    on="user_id",
    how="left"
)

synthetic_ratings_community = synthetic_ratings_comm.rename(columns={
    'user_id': 'UserID',
    'movie_id': 'MovieID'   
})

synthetic_ratings_community.to_csv('synthetic_ratings_community.csv')