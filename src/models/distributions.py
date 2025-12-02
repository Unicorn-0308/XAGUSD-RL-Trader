"""Custom probability distributions for hybrid action spaces."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Distribution


class TruncatedNormal(Distribution):
    """Truncated Normal distribution.
    
    A Normal distribution truncated to lie within [low, high].
    Useful for bounded continuous actions.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """Initialize truncated normal distribution.
        
        Args:
            loc: Mean of the distribution
            scale: Standard deviation
            low: Lower bound
            high: Upper bound
            eps: Small value for numerical stability
        """
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.eps = eps
        
        self._normal = Normal(loc, scale)
        
        batch_shape = loc.shape
        super().__init__(batch_shape=batch_shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the truncated distribution."""
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sample."""
        shape = sample_shape + self.batch_shape
        
        # Sample from uniform and transform
        u = torch.rand(shape, device=self.loc.device)
        
        # CDF at bounds
        cdf_low = self._normal.cdf(torch.tensor(self.low, device=self.loc.device))
        cdf_high = self._normal.cdf(torch.tensor(self.high, device=self.loc.device))
        
        # Inverse CDF (quantile function)
        p = cdf_low + u * (cdf_high - cdf_low)
        p = torch.clamp(p, self.eps, 1 - self.eps)
        
        # Use inverse of standard normal CDF
        sample = self.loc + self.scale * torch.erfinv(2 * p - 1) * (2 ** 0.5)
        
        return torch.clamp(sample, self.low, self.high)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of a value."""
        # Log prob of normal
        log_prob = self._normal.log_prob(value)
        
        # Normalize by the probability mass in [low, high]
        cdf_low = self._normal.cdf(torch.tensor(self.low, device=self.loc.device))
        cdf_high = self._normal.cdf(torch.tensor(self.high, device=self.loc.device))
        log_norm = torch.log(cdf_high - cdf_low + self.eps)
        
        return log_prob - log_norm

    def entropy(self) -> torch.Tensor:
        """Compute entropy (approximate)."""
        # Use normal entropy as approximation
        return self._normal.entropy()

    @property
    def mean(self) -> torch.Tensor:
        """Distribution mean."""
        return self.loc

    @property
    def stddev(self) -> torch.Tensor:
        """Distribution standard deviation."""
        return self.scale


class HybridDistribution:
    """Combined distribution for hybrid action spaces.
    
    Handles both continuous (prediction) and discrete (trading action)
    components together.
    """

    def __init__(
        self,
        continuous_mean: torch.Tensor,
        continuous_std: torch.Tensor,
        discrete_logits: torch.Tensor,
        continuous_low: float | None = None,
        continuous_high: float | None = None,
    ) -> None:
        """Initialize hybrid distribution.
        
        Args:
            continuous_mean: Mean for continuous actions [batch, cont_dim]
            continuous_std: Std for continuous actions [batch, cont_dim]
            discrete_logits: Logits for discrete actions [batch, num_actions]
            continuous_low: Optional lower bound for continuous actions
            continuous_high: Optional upper bound for continuous actions
        """
        self.continuous_dim = continuous_mean.shape[-1]
        self.discrete_dim = discrete_logits.shape[-1]
        
        # Create distributions
        if continuous_low is not None and continuous_high is not None:
            self.continuous_dist = TruncatedNormal(
                continuous_mean, continuous_std,
                low=continuous_low, high=continuous_high,
            )
        else:
            self.continuous_dist = Normal(continuous_mean, continuous_std)
        
        self.discrete_dist = Categorical(logits=discrete_logits)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from both distributions.
        
        Returns:
            Tuple of (continuous_action, discrete_action)
        """
        continuous = self.continuous_dist.sample()
        discrete = self.discrete_dist.sample()
        return continuous, discrete

    def rsample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample (only for continuous part)."""
        if hasattr(self.continuous_dist, "rsample"):
            continuous = self.continuous_dist.rsample()
        else:
            continuous = self.continuous_dist.sample()
        discrete = self.discrete_dist.sample()
        return continuous, discrete

    def log_prob(
        self,
        continuous_action: torch.Tensor,
        discrete_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities.
        
        Args:
            continuous_action: Continuous action values
            discrete_action: Discrete action indices
            
        Returns:
            Tuple of (continuous_log_prob, discrete_log_prob)
        """
        cont_log_prob = self.continuous_dist.log_prob(continuous_action)
        if cont_log_prob.dim() > 1:
            cont_log_prob = cont_log_prob.sum(dim=-1)
        
        disc_log_prob = self.discrete_dist.log_prob(discrete_action)
        
        return cont_log_prob, disc_log_prob

    def entropy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute entropies.
        
        Returns:
            Tuple of (continuous_entropy, discrete_entropy)
        """
        cont_entropy = self.continuous_dist.entropy()
        if cont_entropy.dim() > 1:
            cont_entropy = cont_entropy.sum(dim=-1)
        
        disc_entropy = self.discrete_dist.entropy()
        
        return cont_entropy, disc_entropy

    def combined_entropy(
        self,
        continuous_weight: float = 0.1,
        discrete_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute weighted combined entropy.
        
        Args:
            continuous_weight: Weight for continuous entropy
            discrete_weight: Weight for discrete entropy
            
        Returns:
            Combined entropy value
        """
        cont_ent, disc_ent = self.entropy()
        return continuous_weight * cont_ent + discrete_weight * disc_ent

    @property
    def continuous_mean(self) -> torch.Tensor:
        """Get mean of continuous distribution."""
        return self.continuous_dist.mean

    @property
    def continuous_std(self) -> torch.Tensor:
        """Get std of continuous distribution."""
        return self.continuous_dist.stddev

    @property
    def discrete_probs(self) -> torch.Tensor:
        """Get probabilities of discrete actions."""
        return self.discrete_dist.probs

    def mode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mode of both distributions (deterministic action).
        
        Returns:
            Tuple of (continuous_mode, discrete_mode)
        """
        continuous_mode = self.continuous_dist.mean
        discrete_mode = self.discrete_dist.logits.argmax(dim=-1)
        return continuous_mode, discrete_mode


class SquashedNormal(Distribution):
    """Normal distribution followed by tanh squashing.
    
    Useful for bounded continuous actions in [-1, 1].
    The log_prob is adjusted for the tanh transformation.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Initialize squashed normal distribution.
        
        Args:
            loc: Mean of the underlying normal
            scale: Std of the underlying normal
            eps: Small value for numerical stability
        """
        self.loc = loc
        self.scale = scale
        self.eps = eps
        self._normal = Normal(loc, scale)
        super().__init__(batch_shape=loc.shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the distribution."""
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sample."""
        x = self._normal.rsample(sample_shape)
        return torch.tanh(x)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability with tanh correction.
        
        log p(y) = log p(x) - log |det(dy/dx)|
                 = log p(x) - sum(log(1 - tanh(x)^2))
        """
        # Inverse tanh to get x from y
        value = torch.clamp(value, -1 + self.eps, 1 - self.eps)
        x = 0.5 * (torch.log(1 + value) - torch.log(1 - value))
        
        # Log prob of x under normal
        log_prob = self._normal.log_prob(x)
        
        # Correction for tanh
        log_prob -= torch.log(1 - value.pow(2) + self.eps)
        
        return log_prob

    def entropy(self) -> torch.Tensor:
        """Approximate entropy."""
        # Use normal entropy as approximation
        return self._normal.entropy()

    @property
    def mean(self) -> torch.Tensor:
        """Distribution mean (tanh of normal mean)."""
        return torch.tanh(self.loc)

    @property
    def stddev(self) -> torch.Tensor:
        """Distribution std (approximate)."""
        return self.scale

