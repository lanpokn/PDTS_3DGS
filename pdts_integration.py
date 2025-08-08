# pdts_integration.py

"""
PDTS Integration for 3D Gaussian Splatting (Corrected and Annotated Version)

This module adapts the core concepts from the ICML 2025 paper "Fast and Robust: Task Sampling 
with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments"
by Qu, Wang, et al. for the purpose of accelerating 3DGS training via intelligent view selection.

Key Adaptations:
- "Task" in PDTS is analogous to "View" in 3DGS.
- "Risk" or "Loss" is the 3DGS rendering loss for a specific view.
- "Task Identifier (τ)" is a feature vector extracted from the camera parameters of a view.

This implementation includes critical corrections to align with the PDTS paper's theory,
particularly regarding posterior sampling and diversity-aware selection.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import random

# =====================================================================================
#  SECTION 1: RISK LEARNER ARCHITECTURE
#  This is a faithful implementation of the Neural Process-based architecture
#  described in the PDTS paper (Appendix A.5, Figure 8).
# =====================================================================================

class Encoder(nn.Module):
    """
    Encodes (view_feature, true_loss) pairs into a representation vector r_i.
    Corresponds to the encoder part of a Neural Process.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]
        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)

class MuSigmaEncoder(nn.Module):
    """
    Aggregates a set of representations {r_i} and maps them to the parameters
    (mu, sigma) of the latent variable z's posterior distribution q(z|History).
    """
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Enforce a minimum std deviation for stability
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma

class Decoder(nn.Module):
    """
    Decodes a (view_feature, latent_sample_z) pair to a predicted loss y.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, y_dim)]
        self.xz_to_hidden = nn.Sequential(*layers)

    def forward(self, x, z):
        num_samples, batch_size, _ = z.size()
        # Reshape for batch processing
        x_flat = x.unsqueeze(0).repeat([num_samples, 1, 1, 1]).view(num_samples * batch_size, -1, self.x_dim)
        z_flat = z.view(num_samples * batch_size, self.z_dim).unsqueeze(1).repeat(1, x.shape[1], 1)
        
        input_pairs = torch.cat((x_flat, z_flat), dim=2).view(-1, self.x_dim + self.z_dim)
        
        return self.xz_to_hidden(input_pairs).view(num_samples, batch_size, -1, self.y_dim)

class RiskLearner(nn.Module):
    """
    The complete Neural Process-based model for predicting view-specific rendering loss.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(RiskLearner, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        batch_size, num_points, _ = x.size()
        # Flatten for efficient processing
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        
        # Aggregate the representations from the context set
        r = self.aggregate(r_i)
        return self.r_to_mu_sigma(r)

    def forward(self, context_x, context_y, target_x, num_samples=1):
        """
        Main forward pass. Takes context data to form a posterior, then predicts on target data.
        """
        # Infer posterior q(z|context) from context data
        mu, sigma = self.xy_to_mu_sigma(context_x, context_y)
        z_posterior = Normal(mu, sigma)
        
        # Sample z from the posterior
        z_sample = z_posterior.rsample([num_samples])
        
        # Decode to get predictions for target_x
        y_pred = self.xz_to_y(target_x, z_sample)
        
        return y_pred, z_posterior

# =====================================================================================
#  SECTION 2: TRAINER CLASS 
#  Manages the optimization and prediction logic.
# =====================================================================================
class RiskLearnerTrainer:
    """
    Manages the training and prediction pipeline for the RiskLearner.
    """
    def __init__(self, device, risklearner, optimizer):
        self.device = device
        self.risklearner = risklearner
        self.optimizer = optimizer
        
        # The prior distribution p(z) for the latent variable. It's updated after each training step.
        self.z_prior = Normal(
            torch.zeros([1, self.risklearner.z_dim]).to(self.device),
            torch.ones([1, self.risklearner.z_dim]).to(self.device)
        )
        self.history_x = None
        self.history_y = None

    def train_step(self, context_x, context_y):
        """
        Performs a single training step based on the provided context data (the history).
        This corresponds to maximizing the GELBO in Eq (5) of the PDTS paper.
        """
        # Reshape data to have a batch dimension of 1, as we are in a single-scene setting
        context_x = context_x.unsqueeze(0)
        context_y = context_y.unsqueeze(0).unsqueeze(-1)
        
        self.optimizer.zero_grad()
        
        # During training, we use the context set as the target set to compute reconstruction loss
        y_pred, z_posterior = self.risklearner(context_x, context_y, context_x, num_samples=20)
        
        # 1. Reconstruction Loss (Negative Log-Likelihood term in ELBO)
        # We compare the average prediction with the ground truth
        y_pred_mean = y_pred.mean(dim=0)
        log_likelihood = F.mse_loss(y_pred_mean, context_y, reduction="sum")
        
        # 2. KL Divergence (Regularization term in ELBO)
        # This keeps the posterior close to the prior (which is the posterior from the last step)
        kl = kl_divergence(z_posterior, self.z_prior).mean(dim=0).sum()
        
        # Total loss is the Evidence Lower Bound (ELBO) to be maximized (or minimized as negative ELBO)
        loss = log_likelihood + kl
        
        loss.backward()
        self.optimizer.step()
        
        # Streaming Variational Inference: The current posterior becomes the prior for the next training step.
        self.z_prior = Normal(z_posterior.loc.detach(), z_posterior.scale.detach())
        # Store the history used for this training step, which will form the context for the next prediction
        self.history_x = context_x
        self.history_y = context_y
        
        return loss.item()

    def predict_for_sampling(self, candidate_x):
        """
        Predicts the risk for candidate views using Posterior Sampling.
        This corresponds to Eq (12)a and (12)b in the PDTS paper.
        """
        candidate_x = candidate_x.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            if self.history_x is None:
                # If no history is available (e.g., first prediction), we sample z from the initial prior.
                # Here we pass dummy context data as it won't be used to form a posterior.
                # This part is a simplification; a more robust way would be to just use the decoder with a prior sample.
                z_sample = self.z_prior.rsample([1])
                y_pred = self.risklearner.xz_to_y(candidate_x, z_sample)
            else:
                # The standard case: use the accumulated historical data as context to get an informed posterior.
                y_pred, _ = self.risklearner(
                    self.history_x, 
                    self.history_y, 
                    candidate_x, 
                    num_samples=1 # Critical for Posterior Sampling: only one sample of z is used.
                )
        
        # **CRITICAL CORRECTION**: Per PDTS/Thompson Sampling theory, the acquisition score IS the random sample.
        # We do NOT take the mean. This injects stochasticity, which drives exploration.
        # The shape of y_pred is [num_samples=1, batch_size=1, num_candidates, y_dim=1]
        acquisition_scores = y_pred.squeeze() # Resulting shape: [num_candidates]
        return acquisition_scores

# =====================================================================================
#  SECTION 3: DIVERSIFIED SAMPLING (Corrected and Robust Implementation)
# =====================================================================================

def msd_diversified_selection(candidates_features, acquisition_scores, lambda_diversity, num_selected):
    """
    Selects a subset of views by greedily maximizing a score that combines
    acquisition (predicted loss) and diversity (distance in feature space).
    **This is the corrected implementation with normalization.**
    """
    if len(candidates_features) <= num_selected:
        return list(range(len(candidates_features)))

    candidates_np = candidates_features.cpu().numpy()
    acquisition_scores_np = acquisition_scores.cpu().numpy()

    # --- Step 1: Normalize acquisition scores to be on a comparable scale [0, 1] ---
    # This is the CRITICAL fix. It prevents one term from dominating the other.
    acq_min, acq_max = acquisition_scores_np.min(), acquisition_scores_np.max()
    # Add a small epsilon to prevent division by zero if all scores are the same
    acq_normalized = (acquisition_scores_np - acq_min) / (acq_max - acq_min + 1e-8)

    selected_indices = []
    remaining_indices = list(range(len(candidates_np)))

    # Greedily select num_selected items
    for i in range(num_selected):
        if i == 0:
            # First selection is purely based on the highest acquisition score
            best_idx_local = np.argmax(acq_normalized[remaining_indices])
            best_idx_global = remaining_indices[best_idx_local]
        else:
            best_score = -np.inf
            best_idx_global = -1
            
            # --- Step 2: Calculate diversity scores for all remaining candidates ---
            selected_features = candidates_np[selected_indices]
            remaining_features = candidates_np[remaining_indices]
            
            # For each remaining point, find its minimum distance to the already selected set
            distances = cdist(remaining_features, selected_features, metric='euclidean')
            min_distances = np.min(distances, axis=1)
            
            # Normalize diversity scores to [0, 1] as well
            dist_min, dist_max = min_distances.min(), min_distances.max()
            div_normalized = (min_distances - dist_min) / (dist_max - dist_min + 1e-8)
            
            # --- Step 3: Combine normalized scores with a clear trade-off parameter `lambda` ---
            # This is the correct way to balance two objectives.
            remaining_acq_normalized = acq_normalized[remaining_indices]
            combined_scores = (1 - lambda_diversity) * remaining_acq_normalized + lambda_diversity * div_normalized
            
            best_idx_local = np.argmax(combined_scores)
            best_idx_global = remaining_indices[best_idx_local]

        selected_indices.append(best_idx_global)
        remaining_indices.remove(best_idx_global)
        
    return selected_indices

# =====================================================================================
#  SECTION 4: MAIN SELECTOR CLASS AND FEATURE EXTRACTION
# =====================================================================================

def extract_camera_features(camera):
    """
    Extracts a feature vector from a 3DGS Camera object.
    **IMPROVED**: Uses 6D rotation representation instead of Euler angles for better stability
    and continuity, which is crucial for the neural network to learn effectively.
    """
    features = []
    # 3D: Camera position in world coordinates
    features.extend(camera.camera_center.cpu().numpy().tolist())
    
    # 6D: Continuous 6D representation for rotation, avoiding gimbal lock
    R = camera.R
    features.extend(R[:2, :].flatten().tolist())
    
    # 2D: Field of View in x and y
    features.extend([camera.FoVx, camera.FoVy])
    
    # 2D: Image resolution
    features.extend([camera.image_width, camera.image_height])
    
    # Total dimension is now 3 + 6 + 2 + 2 = 13
    return torch.tensor(features, dtype=torch.float32)

class PDTSViewSelector:
    """
    The main interface that integrates the RiskLearner with the 3DGS training loop.
    
    NEW STRATEGY: Hybrid Network + Random Selection
    This version implements a robust hybrid selection strategy. For each batch of views,
    a portion is selected by the PDTS network (exploitation), and the rest is
    selected completely randomly (exploration). This removes the need for complex
    bootstrap and recovery period management, making the process more stable and
    resilient to abrupt changes in the loss landscape (e.g., after opacity resets).
    """
    def __init__(self, device, x_dim=13, lambda_diversity=0.5, network_ratio=2/3, bootstrap_iterations=1000):
        # NEW: Added dynamic network_ratio strategy with bootstrap phase
        self.device = device
        self.x_dim = x_dim
        self.lambda_diversity = lambda_diversity
        self.network_ratio = network_ratio
        self.bootstrap_iterations = bootstrap_iterations
        
        self.training_data_x = []
        self.training_data_y = []
        
        self.risklearner = RiskLearner(x_dim=self.x_dim, y_dim=1, r_dim=10, z_dim=10, h_dim=10).to(device)
        self.optimizer = torch.optim.Adam(self.risklearner.parameters(), lr=0.001)
        self.trainer = RiskLearnerTrainer(device, self.risklearner, self.optimizer)
        
        print(f"[PDTS] Initialized with Dynamic Hybrid Strategy.")
        print(f"[PDTS] Bootstrap Phase: 0-{bootstrap_iterations} iterations (100% Random)")
        print(f"[PDTS] Hybrid Phase: {bootstrap_iterations}+ iterations ({self.network_ratio*100:.1f}% Network, {(1-self.network_ratio)*100:.1f}% Random)")

    # REMOVED: All old state-management methods are gone.
    # _initialize_optimizer_and_trainer, _reset_for_new_epoch, 
    # check_and_perform_reset, _is_in_recovery_period, should_use_network

    def add_training_data(self, camera, loss_value, current_iteration):
        """Stores a new {view, loss} pair for training the RiskLearner (only after bootstrap)."""
        # Skip data collection during bootstrap phase
        if current_iteration <= self.bootstrap_iterations:
            return  # Don't collect any data during bootstrap
        
        features = extract_camera_features(camera)
        self.training_data_x.append(features)
        self.training_data_y.append(loss_value)
        
        # Simple memory management
        max_data_size = 1000
        if len(self.training_data_x) > max_data_size:
            self.training_data_x.pop(0)
            self.training_data_y.pop(0)

    def train_network(self, current_iteration):
        """Periodically trains the RiskLearner on the collected historical data (only after bootstrap)."""
        # Skip training during bootstrap phase
        if current_iteration <= self.bootstrap_iterations:
            return None  # Don't train during bootstrap
            
        if len(self.training_data_x) < 20:
            return None
        x_data = torch.stack(self.training_data_x).to(self.device)
        y_data = torch.tensor(self.training_data_y, dtype=torch.float32).to(self.device)
        loss = self.trainer.train_step(x_data, y_data)
        return loss

    # COMPLETELY REWRITTEN: select_views with the new dynamic hybrid logic
    def select_views(self, current_iteration, viewpoint_pool, num_selected=16, num_candidates=256):
        """
        Selects views using dynamic strategy:
        - Before bootstrap_iterations: Pure random (network_ratio = 0)
        - After bootstrap_iterations: Hybrid selection (network_ratio = configured value)
        """
        # Dynamic network ratio based on iteration
        if current_iteration <= self.bootstrap_iterations:
            effective_network_ratio = 0.0  # Pure random during bootstrap
        else:
            effective_network_ratio = self.network_ratio  # Use configured ratio after bootstrap
        
        # Fallback to pure random if not enough data to even try using the network.
        if len(self.training_data_x) < 20 or effective_network_ratio == 0.0:
            selected_indices = random.sample(range(len(viewpoint_pool)), min(num_selected, len(viewpoint_pool)))
            phase = "bootstrap" if current_iteration <= self.bootstrap_iterations else "random_fallback"
            return [viewpoint_pool[i] for i in selected_indices], phase

        try:
            # --- Step 1: Calculate the number of views for each strategy ---
            num_network = round(num_selected * effective_network_ratio)
            num_random = num_selected - num_network
            
            # --- Step 2: Create a candidate pool for selection ---
            num_candidates = min(num_candidates, len(viewpoint_pool))
            candidate_indices_pool = random.sample(range(len(viewpoint_pool)), num_candidates)
            
            # --- Step 3: Select views using the PDTS Network ---
            candidate_cameras = [viewpoint_pool[i] for i in candidate_indices_pool]
            candidate_features = torch.stack([extract_camera_features(cam) for cam in candidate_cameras])
            acquisition_scores = self.trainer.predict_for_sampling(candidate_features)
            
            # Ask for `num_network` views from the diversified selection algorithm
            network_selected_local_indices = msd_diversified_selection(
                candidate_features, acquisition_scores, self.lambda_diversity, num_network
            )
            # Map local candidate indices back to global viewpoint_pool indices
            network_final_indices = {candidate_indices_pool[i] for i in network_selected_local_indices}

            # --- Step 4: Select remaining views randomly, ensuring no overlap ---
            # Create a pool of indices that were NOT selected by the network
            remaining_global_indices = [idx for idx in range(len(viewpoint_pool)) if idx not in network_final_indices]
            
            # Randomly sample `num_random` views from the remaining pool
            if len(remaining_global_indices) >= num_random:
                random_final_indices = set(random.sample(remaining_global_indices, num_random))
            else:
                # If not enough remaining views, just take all of them
                random_final_indices = set(remaining_global_indices)

            # --- Step 5: Combine the selections ---
            final_indices = list(network_final_indices.union(random_final_indices))
            
            return [viewpoint_pool[i] for i in final_indices], "hybrid_network"
            
        except Exception as e:
            # A robust fallback for any error during network selection
            print(f"[PDTS] CRITICAL ERROR during hybrid selection: {e}. Falling back to pure random.")
            selected_indices = random.sample(range(len(viewpoint_pool)), min(num_selected, len(viewpoint_pool)))
            return [viewpoint_pool[i] for i in selected_indices], "fallback_random"