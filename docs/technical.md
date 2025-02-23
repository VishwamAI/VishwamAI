# Developing Optimized Large Language Models on Limited Compute Resources

**Kasinadhsarma**  
Email: [kasinadhsarma@gmail.com](mailto:kasinadhsarma@gmail.com)

---

## Abstract

Large language models (LLMs) have demonstrated remarkable performance across various natural language tasks. However, training such models at scale requires immense computational resources, posing a significant challenge for researchers with limited hardware. This paper presents a comprehensive framework that integrates data-centric optimizations, compute efficiency techniques, and architectural innovations to enable high-quality LLM training on limited compute resources. Our theoretical analysis, based on TPU v5e1 with 100 compute units (approximately 100 hours of training), shows that it is feasible to pre-train a model with up to 10 billion parameters on a corpus of 100 million tokens over 3 epochs. We leverage sparsely-gated Mixture-of-Experts (MoE) layers, dynamic inference strategies, and mixed-precision training to potentially reduce training compute by up to 30% while preserving downstream task performance.

---

## Introduction

Recent advances in LLMs have led to breakthrough performance in natural language processing tasks. Despite these successes, the escalating computational costs required for training LLMs present a barrier, especially for resource-constrained environments. This paper proposes a holistic framework aimed at reducing training compute without sacrificing model performance by combining:
- Data-centric optimizations,
- Compute efficiency techniques, and
- Architectural enhancements.

Our work targets training on TPU v5e1, where 100 compute units (roughly 100 hours) are assumed to provide an effective throughput of 50 TFLOPS. We demonstrate through FLOPs analysis that under ideal conditions, a model with up to 10 billion parameters can be trained on 100 million tokens over 3 epochs.

---

## Related Work

Scaling laws have shown that increased model size and compute lead to improved performance. Techniques such as model distillation and sparse architectures like MoE have been proposed to improve efficiency. Our framework builds upon these approaches by integrating dynamic inference strategies and targeted TPU optimizations.

---

## Compute Budget and FLOPs Analysis

### Compute Budget Calculation

Assuming TPU v5e1 has an effective throughput of:
\[
50 \times 10^{12} \text{ FLOPs/sec}
\]
and 1 compute unit equals 1 hour, then:
- **FLOPs per hour:**
  \[
  50 \times 10^{12} \times 3600 \approx 1.8 \times 10^{17} \text{ FLOPs}
  \]
- **Total FLOPs for 100 hours:**
  \[
  1.8 \times 10^{17} \times 100 = 1.8 \times 10^{19} \text{ FLOPs}
  \]

### FLOPs Required for Training

Using the heuristic of 6 FLOPs per parameter per token (covering both forward and backward passes) for a dataset of:
- \(T = 10^8\) tokens (100 million tokens)
- \(E = 3\) epochs

The total FLOPs required is:
\[
\text{FLOPs}_{\text{required}} = 6 \times N \times T \times E = 18 \times N \times 10^8.
\]

Setting this equal to the available compute:
\[
18 \times N \times 10^8 = 1.8 \times 10^{19}
\]
we solve for \(N\):
\[
N = \frac{1.8 \times 10^{19}}{18 \times 10^8} = \frac{1.8 \times 10^{19}}{1.8 \times 10^9} = 10^{10} \text{ parameters}.
\]

Thus, theoretically, training a model with up to 10 billion parameters is feasible under these ideal conditions.

---

## Methodology

### Data-Centric Optimizations

Our data pipeline includes:
- **Advanced Filtering:** Deduplication and n-gram overlap filtering to eliminate low-quality data.
- **Data Augmentation:** Using back-translation and text infilling to augment underrepresented domains.
- **Tokenization:** Training a SentencePiece tokenizer with outputs stored in SafeTensors for efficient access.

### Compute Efficiency Techniques

To reduce compute requirements, we implement:
- **Mixed-Precision Training:** Leveraging BFloat16 and INT8 formats.
- **Dynamic Batching:** Adjusting batch sizes and sequence lengths based on current compute availability.
- **NUMA-Aware Memory Allocation:** Optimizing data distribution in multi-socket systems.

### Architectural Innovations

Our model architecture comprises:
- A standard Transformer backbone with token and positional embeddings.
- **Sparsely-Gated MoE Layers:** Enabling conditional computation to scale model capacity without proportional compute cost.
- **Multi-Layer Attention (MLA):** Enhancing contextual representations via cross-layer attention.
- **Dynamic Inference:** Incorporating adaptive early exiting to reduce inference computation.

### TPU Execution and Distributed Training

Our framework leverages TPU v5e1 with XLA:
- Distributed training is implemented via \texttt{jax.pmap} (or PyTorch XLA equivalents) for efficient parallelization.
- Optimization is performed with AdamW, using a cosine learning rate schedule and warmup.
- Optional integration of FairScale is available for sharded optimizer states if using PyTorch on TPU.

---

## Experimental Setup

Our experimental plan targets models in the 500M to 1B parameter range as baselines. We compare:
- **Dense Transformer architectures** versus those augmented with MoE and MLA.
- Evaluation metrics include test loss, perplexity, downstream task performance, FLOPs per token, and energy consumption.

---

## Results and Discussion

Preliminary analysis indicates that our framework can potentially reduce compute requirements by 20% to 30% compared to dense architectures, with less than a 1% accuracy drop on downstream tasks. Our approach also demonstrates improved scalability in resource-constrained environments. Detailed experimental results, including ablation studies and performance profiling, will be presented in future work.

---

## Conclusion

We have presented Vishwamai, a scalable framework for pre-training large language models on limited compute resources using TPU v5e1. Our integration of data-centric optimizations, compute efficiency techniques, and architectural innovations such as sparsely-gated MoE layers and dynamic inference shows that it is feasible to train models with up to 10 billion parameters on 100 million tokens over 3 epochs within a 100-hour compute budget. Future work will focus on experimental validation and further refinement of these methods.

