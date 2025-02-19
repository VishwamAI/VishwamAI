import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

class EthicalPrinciple(Enum):
    BENEFICENCE = auto()      # Do good
    NON_MALEFICENCE = auto()  # Avoid harm
    AUTONOMY = auto()         # Respect independence
    JUSTICE = auto()          # Be fair
    TRANSPARENCY = auto()     # Be explainable
    PRIVACY = auto()          # Protect data

@dataclass
class EthicalConfig:
    """Configuration for ethical framework."""
    min_ethical_score: float = 0.7
    privacy_threshold: float = 0.8
    fairness_threshold: float = 0.9
    transparency_level: float = 0.8
    uncertainty_threshold: float = 0.2
    intervention_threshold: float = 0.6
    ethical_memory_size: int = 1000
    principle_weights: Optional[Dict[EthicalPrinciple, float]] = None

class EthicalFramework(nn.Module):
    """
    Framework for ethical decision making and value alignment.
    
    Implements:
    - Ethical principle evaluation
    - Value alignment checking
    - Decision transparency
    - Fairness monitoring
    - Privacy protection
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[EthicalConfig] = None
    ):
        super().__init__()
        self.config = config or EthicalConfig()
        self.hidden_dim = hidden_dim
        
        # Initialize principle weights if not provided
        if self.config.principle_weights is None:
            self.config.principle_weights = {
                principle: 1.0 / len(EthicalPrinciple)
                for principle in EthicalPrinciple
            }
        
        # Ethical evaluation networks
        self.ethical_evaluator = nn.ModuleDict({
            principle.name: self._create_principle_evaluator()
            for principle in EthicalPrinciple
        })
        
        self.value_alignment_checker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(EthicalPrinciple)),
            nn.Sigmoid()
        )
        
        self.explanation_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Monitoring and tracking
        self.decision_history = []
        self.ethical_violations = []
        self.intervention_history = []
        
    def _create_principle_evaluator(self) -> nn.Module:
        """Create a neural network for evaluating an ethical principle."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        decision_state: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Evaluate ethical implications of a decision state.
        
        Args:
            decision_state: Current decision state tensor
            context: Optional context information
            
        Returns:
            Tuple of (modified state, ethical metrics)
        """
        batch_size = decision_state.size(0)
        
        # Evaluate each ethical principle
        principle_scores = {}
        for principle in EthicalPrinciple:
            score = self.ethical_evaluator[principle.name](decision_state)
            principle_scores[principle.name] = score
            
        # Check value alignment
        alignment_scores = self.value_alignment_checker(decision_state)
        
        # Generate explanation if transparency is needed
        explanation = None
        if self._needs_explanation(principle_scores):
            explanation_context = torch.cat([
                decision_state,
                context if context is not None else torch.zeros_like(decision_state)
            ], dim=-1)
            explanation = self.explanation_generator(explanation_context)
            
        # Calculate overall ethical score
        weighted_scores = [
            score.mean() * self.config.principle_weights[principle]
            for principle, score in zip(EthicalPrinciple, principle_scores.values())
        ]
        ethical_score = torch.stack(weighted_scores).sum()
        
        # Track decision
        self._track_decision(ethical_score, principle_scores)
        
        # Check if intervention is needed
        needs_intervention = ethical_score < self.config.intervention_threshold
        if needs_intervention:
            decision_state = self._intervene(decision_state, principle_scores)
            self.intervention_history.append({
                'state': decision_state.detach(),
                'scores': {k: v.item() for k, v in principle_scores.items()},
                'step': len(self.decision_history)
            })
            
        metrics = {
            'ethical_score': ethical_score.item(),
            'principle_scores': {k: v.mean().item() for k, v in principle_scores.items()},
            'alignment_score': alignment_scores.mean().item(),
            'needs_intervention': needs_intervention,
            'has_explanation': explanation is not None,
            'num_violations': len(self.ethical_violations)
        }
        
        return decision_state, metrics
    
    def _needs_explanation(self, principle_scores: Dict[str, torch.Tensor]) -> bool:
        """Determine if decision needs explanation based on principles."""
        if principle_scores['TRANSPARENCY'].mean() < self.config.transparency_level:
            return True
            
        # Check if any principle score is concerning
        return any(
            score.mean() < self.config.min_ethical_score
            for score in principle_scores.values()
        )
    
    def _track_decision(
        self,
        ethical_score: torch.Tensor,
        principle_scores: Dict[str, torch.Tensor]
    ):
        """Track decision and any ethical violations."""
        self.decision_history.append({
            'ethical_score': ethical_score.item(),
            'principle_scores': {k: v.mean().item() for k, v in principle_scores.items()}
        })
        
        # Check for violations
        if ethical_score < self.config.min_ethical_score:
            violation = {
                'step': len(self.decision_history),
                'ethical_score': ethical_score.item(),
                'principle_scores': {k: v.mean().item() for k, v in principle_scores.items()}
            }
            self.ethical_violations.append(violation)
    
    def _intervene(
        self,
        state: torch.Tensor,
        principle_scores: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Modify state to align with ethical principles."""
        # Find most violated principles
        violations = [
            (principle, score.mean())
            for principle, score in principle_scores.items()
            if score.mean() < self.config.min_ethical_score
        ]
        
        if not violations:
            return state
            
        # Sort by severity of violation
        violations.sort(key=lambda x: x[1])
        
        # Apply corrective adjustments for each violated principle
        modified_state = state
        for principle, score in violations:
            correction_network = self.ethical_evaluator[principle]
            correction = correction_network(modified_state)
            modified_state = modified_state + (correction * (1.0 - score))
            
        return modified_state
    
    def get_ethical_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ethical metrics."""
        if not self.decision_history:
            return {}
            
        recent_decisions = self.decision_history[-100:]
        metrics = {
            'avg_ethical_score': np.mean([d['ethical_score'] for d in recent_decisions]),
            'min_ethical_score': min(d['ethical_score'] for d in recent_decisions),
            'violation_rate': len(self.ethical_violations) / len(self.decision_history),
            'intervention_rate': len(self.intervention_history) / len(self.decision_history),
            'principle_trends': {
                principle.name: np.mean([
                    d['principle_scores'][principle.name]
                    for d in recent_decisions
                ])
                for principle in EthicalPrinciple
            }
        }
        return metrics
    
    def explain_decision(
        self,
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate explanation for a decision state."""
        with torch.no_grad():
            explanation_context = torch.cat([
                state,
                context if context is not None else torch.zeros_like(state)
            ], dim=-1)
            
            explanation = self.explanation_generator(explanation_context)
            
            # Get principle-specific explanations
            principle_explanations = {
                principle.name: {
                    'score': self.ethical_evaluator[principle.name](state).item(),
                    'contribution': self._get_principle_contribution(principle, state)
                }
                for principle in EthicalPrinciple
            }
            
            return {
                'general_explanation': explanation,
                'principle_explanations': principle_explanations,
                'confidence': self._calculate_explanation_confidence(state)
            }
    
    def _get_principle_contribution(
        self,
        principle: EthicalPrinciple,
        state: torch.Tensor
    ) -> float:
        """Calculate how much a principle contributed to the decision."""
        with torch.no_grad():
            base_score = self.ethical_evaluator[principle.name](state)
            masked_state = state * 0.0  # Ablation test
            masked_score = self.ethical_evaluator[principle.name](masked_state)
            return (base_score - masked_score).item()
    
    def _calculate_explanation_confidence(self, state: torch.Tensor) -> float:
        """Calculate confidence in the generated explanation."""
        with torch.no_grad():
            scores = [
                self.ethical_evaluator[p.name](state).item()
                for p in EthicalPrinciple
            ]
            score_std = np.std(scores)
            confidence = 1.0 - min(score_std / self.config.uncertainty_threshold, 1.0)
            return float(confidence)
    
    def reset_tracking(self):
        """Reset all tracking history."""
        self.decision_history = []
        self.ethical_violations = []
        self.intervention_history = []
    
    def integrate_with_trainer(self, trainer):
        """Integrate ethical framework with advanced trainer."""
        self.trainer = trainer
        self.trainer.ethical_framework = self

    def update_forward(self, decision_state: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Updated forward method to include new advancements.
        
        Args:
            decision_state: Current decision state tensor
            context: Optional context information
            
        Returns:
            Tuple of (modified state, ethical metrics)
        """
        batch_size = decision_state.size(0)
        
        # Evaluate each ethical principle
        principle_scores = {}
        for principle in EthicalPrinciple:
            score = self.ethical_evaluator[principle.name](decision_state)
            principle_scores[principle.name] = score
            
        # Check value alignment
        alignment_scores = self.value_alignment_checker(decision_state)
        
        # Generate explanation if transparency is needed
        explanation = None
        if self._needs_explanation(principle_scores):
            explanation_context = torch.cat([
                decision_state,
                context if context is not None else torch.zeros_like(decision_state)
            ], dim=-1)
            explanation = self.explanation_generator(explanation_context)
            
        # Calculate overall ethical score
        weighted_scores = [
            score.mean() * self.config.principle_weights[principle]
            for principle, score in zip(EthicalPrinciple, principle_scores.values())
        ]
        ethical_score = torch.stack(weighted_scores).sum()
        
        # Track decision
        self._track_decision(ethical_score, principle_scores)
        
        # Check if intervention is needed
        needs_intervention = ethical_score < self.config.intervention_threshold
        if needs_intervention:
            decision_state = self._intervene(decision_state, principle_scores)
            self.intervention_history.append({
                'state': decision_state.detach(),
                'scores': {k: v.item() for k, v in principle_scores.items()},
                'step': len(self.decision_history)
            })
            
        metrics = {
            'ethical_score': ethical_score.item(),
            'principle_scores': {k: v.mean().item() for k, v in principle_scores.items()},
            'alignment_score': alignment_scores.mean().item(),
            'needs_intervention': needs_intervention,
            'has_explanation': explanation is not None,
            'num_violations': len(self.ethical_violations)
        }
        
        return decision_state, metrics
