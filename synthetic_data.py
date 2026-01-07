import pandas as pd 
import numpy as np
from scipy.special import expit

# Generate synthetic data with latent factors

def simulate_rating_matrix(
    n_users=1000,
    n_movies=5_000,
    n_factors=10,
    user_factor_std=1.0,
    movie_factor_std=1.0,
    random_state=12
):
    rng = np.random.default_rng(random_state)

    # Generate latent factors for users and movies
    U = rng.normal(0, user_factor_std, size=(n_users, n_factors))
    V = rng.normal(0, movie_factor_std, size=(n_movies, n_factors))

    # Compute affinity scores
    scores = U @ V.T  # shape (n_users, n_moves)
    ratings = 1 + 4 * expit(scores)  # transform scores to [1,5]
    ratings = np.round(ratings).astype(int)
   
    
    # Round to nearest integer to get ratings 1-5

    binary_ratings = (ratings >= 4).astype(int)  # Optional: binary ratings for certain applications

    # Create DataFrame for user factors
    user_factors_df = pd.DataFrame(U, columns=[f"factor_{k}" for k in range(n_factors)])
    user_factors_df.insert(0, "user_id", np.arange(n_users))

    # Create DataFrame for movie factors
    movie_factors_df = pd.DataFrame(V, columns=[f"factor_{k}" for k in range(n_factors)])
    movie_factors_df.insert(0, "movie_id", np.arange(n_movies))

    # Create interactions DataFrame in long format
    user_ids = np.repeat(np.arange(n_users), n_movies)
    movie_ids = np.tile(np.arange(n_movies), n_users)

    interactions_df = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings.ravel(),
        "binary_ratings" : binary_ratings.ravel()
    })

    return ratings, user_factors_df, movie_factors_df, interactions_df

ratings, user_factors_df, movie_factors_df, interactions_df = simulate_rating_matrix()


# Rename columns to match expected format
synthetic_ratings = interactions_df.rename(columns={
    'user_id': 'UserID',
    'movie_id': 'MovieID',
    'binary_ratings': 'binary_rating',
    'rating': 'rating'    
})
synthetic_ratings.to_csv('synthetic_ratings.csv')

