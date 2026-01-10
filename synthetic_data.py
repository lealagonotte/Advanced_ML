import pandas as pd 
import numpy as np
from scipy.special import expit

# Generate synthetic data with latent factors

import numpy as np
import pandas as pd
from scipy.special import expit

def simulate_rating_matrix(
    n_users=1000,
    n_movies=5_000,
    n_factors=10,
    user_factor_std=1.0,
    movie_factor_std=1.0,
    random_state=12
):
    rng = np.random.default_rng(random_state)

    # 1. Generate latent factors
    U = rng.normal(0, user_factor_std, size=(n_users, n_factors))
    V = rng.normal(0, movie_factor_std, size=(n_movies, n_factors))

    # 2. Compute affinity scores (Latent preference)
    scores = U @ V.T  # shape (n_users, n_movies)
    
    # 3. Transform scores to probabilities [0, 1] using sigmoid
    # expit is equivalent to 1 / (1 + exp(-x))
    probs = expit(scores)
    
    # 4. Generate Noisy Ratings using Shifted Binomial Distribution
    # We sample from B(n=4, p) -> gives integers in [0, 4]
    # Then add 1 -> gives integers in [1, 5]
    # This adds stochasticity: a user with high probability p=0.9 can still rate 4 instead of 5.
    raw_ratings = rng.binomial(n=4, p=probs)
    ratings = 1 + raw_ratings
    
    # 5. Create Binary Ratings (Standard approach: >= 4 is positive)
    binary_ratings = (ratings >= 4).astype(int)

    # --- Formatting Outputs (Same as before) ---
    
    user_factors_df = pd.DataFrame(U, columns=[f"factor_{k}" for k in range(n_factors)])
    user_factors_df.insert(0, "user_id", np.arange(n_users))

    movie_factors_df = pd.DataFrame(V, columns=[f"factor_{k}" for k in range(n_factors)])
    movie_factors_df.insert(0, "movie_id", np.arange(n_movies))

    # Using ravel() to flatten the matrices for the DataFrame
    # Note: Creating a full dense DataFrame for 1000x5000 = 5M rows is heavy but manageable.
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


# Rename columns to match expected format
synthetic_ratings = interactions_df.rename(columns={
    'user_id': 'UserID',
    'movie_id': 'MovieID',
    'binary_ratings': 'binary_rating',
    'rating': 'rating'    
})
synthetic_ratings.to_csv('synthetic_ratings.csv')

