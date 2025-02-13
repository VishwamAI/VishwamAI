import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .structured_reasoning import StructuredReasoning
from .verification import SelfVerification 
from .metacognition import MetaCognition
from .deepthinking import THINK_START, THINK_END, REFLECT_START, REFLECT_END, STEP_START, STEP_END

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    premise: torch.Tensor
    inference: torch.Tensor
    confidence: float
    step_num: int
    reflection: Optional[str] = None

@dataclass
class ReasoningOutput:
    """Enhanced reasoning output with structured steps and reflections"""
    steps: List[ReasoningStep]
    verification: Dict[str, torch.Tensor]
    meta_analysis: Dict[str, torch.Tensor]
    reflections: List[Dict[str, str]]
    final_answer: str
    confidence: float
    trace: Dict[str, List[float]]

class EnhancedReasoning(nn.Module):
    """Advanced reasoning module with self-reflection and step tracking"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Core reasoning components
        self.structured_reasoning = StructuredReasoning(hidden_size)
        self.verification = SelfVerification(hidden_size)
        self.metacognition = MetaCognition(hidden_size)
        
        # New components for enhanced reasoning
        self.step_encoder = nn.Linear(hidden_size, hidden_size)
        self.reflection_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Integration layers
        self.reasoning_merger = nn.Linear(hidden_size * 3, hidden_size)
        self.final_classifier = nn.Linear(hidden_size, config.vocab_size)
        
        # Confidence scoring
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def generate_reflection(
        self,
        current_state: torch.Tensor,
        reasoning_history: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Generate a reflection on the current reasoning state"""
        # Ensure tensors are on the same device as the model
        current_state = current_state.to(self.config.device)
        reasoning_history = reasoning_history.to(self.config.device)
        combined = torch.cat([current_state, reasoning_history], dim=-1)
        
        reflection_repr = self.reflection_generator(combined)
        confidence_score = self.confidence_scorer(combined)
        
        # Properly reduce dimensions for confidence
        if len(confidence_score.shape) > 1:
            confidence = confidence_score.mean(dim=tuple(range(len(confidence_score.shape)-1)))
        else:
            confidence = confidence_score.mean()
            
        return reflection_repr, float(confidence.item())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        knowledge_context: Optional[torch.Tensor] = None,
        max_steps: int = 5,
        reflection_threshold: float = 0.7
    ) -> ReasoningOutput:
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Initialize tracking
        all_steps = []
        all_reflections = []
        confidence_trace = []
        reasoning_history = torch.zeros_like(hidden_states)
        
        for step in range(max_steps):
            # Generate reasoning step
            current_step = self.structured_reasoning(
                hidden_states,
                attention_mask,
                step_context=self.step_encoder(reasoning_history)
            )
            
            # Track step
            all_steps.append(ReasoningStep(
                premise=current_step.premise,
                inference=current_step.inference,
                confidence=current_step.confidence,
                step_num=step + 1
            ))
            
            confidence_trace.append(current_step.confidence)
            
            # Update reasoning history
            reasoning_history = reasoning_history + current_step.inference
            
            # Generate reflection if confidence is below threshold
            # For low threshold, generate more reflections
            if (1.0 - current_step.confidence) > reflection_threshold:
                reflection_repr, ref_confidence = self.generate_reflection(
                    current_step.inference,
                    reasoning_history
                )
                
                all_reflections.append({
                    "step": step + 1,
                    "reflection": reflection_repr.tolist(),
                    "confidence": ref_confidence
                })
                
                # Update reasoning with reflection insights
                reasoning_history = reasoning_history + reflection_repr
            
            # Check if we've reached sufficient confidence
            if current_step.confidence > 0.95:
                break
        
        # Verify reasoning
        # Get verification results
        verification_context = hidden_states if knowledge_context is None else knowledge_context
        verification_results = {
            **self.verification(hidden_states, verification_context, reasoning_history),
            'context': verification_context,
            'scores': self.verification(hidden_states, verification_context, reasoning_history)['factual_validity']
        }
        
        # Metacognitive analysis
        meta_results = self.metacognition(
            hidden_states,
            torch.stack([step.inference for step in all_steps]).view(batch_size, -1, hidden_states.size(-1))
        )
        
        # Integrate all components for final answer
        # Ensure all tensors have the same shape before concatenation
        final_repr = self.reasoning_merger(torch.cat([
            hidden_states,
            reasoning_history,
            all_steps[-1].inference.view(batch_size, -1, hidden_states.size(-1))
        ], dim=-1))
        
        logits = self.final_classifier(final_repr)
        
        return ReasoningOutput(
            steps=all_steps,
            verification=verification_results,
            meta_analysis=meta_results,
            reflections=all_reflections,
            final_answer=logits.argmax(-1).tolist(),
            confidence=meta_results["confidence"].mean().item(),
            trace={
                "step_confidence": [float(c) for c in confidence_trace],
                "verification_scores": [float(s) for s in verification_results["scores"].flatten()],
                "meta_confidence": [float(c) for c in meta_results["confidence"].flatten()]
            }
        )

    def analyze_reasoning_trace(self, output: ReasoningOutput) -> Dict[str, float]:
        """Analyze the reasoning process for quality metrics"""
        try:
            # Convert lists to tensors for proper numerical operations
            step_confidence = torch.tensor(output.trace["step_confidence"], device=self.config.device)
            verification_scores = torch.tensor(output.trace["verification_scores"], device=self.config.device)
            meta_confidence = torch.tensor(output.trace["meta_confidence"], device=self.config.device)
            
            # Calculate metrics using tensor operations
            return {
                "avg_step_confidence": float(step_confidence.mean().item()),
                "verification_score": float(verification_scores.mean().item()),
                "meta_confidence": float(meta_confidence.mean().item()),
                "num_reflections": float(len(output.reflections)),
                "final_confidence": float(output.confidence)
            }
        except Exception:
            # Fallback to basic list operations if tensor operations fail
            return {
                "avg_step_confidence": sum(output.trace["step_confidence"]) / max(len(output.trace["step_confidence"]), 1),
                "verification_score": sum(output.trace["verification_scores"]) / max(len(output.trace["verification_scores"]), 1),
                "meta_confidence": sum(output.trace["meta_confidence"]) / max(len(output.trace["meta_confidence"]), 1),
                "num_reflections": float(len(output.reflections)),
                "final_confidence": float(output.confidence)
            }

    def format_reasoning_output(self, output: ReasoningOutput) -> str:
        """Format the reasoning process in a structured way with CoT markers"""
        formatted = []
        
        # Start thinking process
        formatted.append(f"{THINK_START}")
        
        # Add each reasoning step
        for step in output.steps:
            formatted.append(f"{STEP_START}Step {step.step_num}:")
            formatted.append(f"Premise: {step.premise}")
            formatted.append(f"Inference: {step.inference}")
            formatted.append(f"Confidence: {step.confidence:.2f}{STEP_END}")
            
            # Add reflection if available
            matching_reflections = [r for r in output.reflections if r["step"] == step.step_num]
            if matching_reflections:
                formatted.append(f"{REFLECT_START}")
                for reflection in matching_reflections:
                    formatted.append(f"Reflection: {reflection['reflection']}")
                    formatted.append(f"Confidence after reflection: {reflection['confidence']:.2f}")
                formatted.append(f"{REFLECT_END}")
        
        formatted.append(f"{THINK_END}")
        
        # Add final answer
        formatted.append(f"Final Answer: {output.final_answer}")
        formatted.append(f"Overall Confidence: {output.confidence:.2f}")
        
        return "\n".join(formatted)
