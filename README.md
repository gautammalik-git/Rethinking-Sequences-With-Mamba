# Rethinking-Sequences-With-Mamba

## 🐍 My Fascination with Mamba (and Why You Should Care)

In the world of deep learning, **Transformers** have become the go-to architecture for almost every sequence modeling task, not just because they perform well, but because they're everywhere. If you want to solve a problem and get great results, using a Transformer is often the safest bet.

But for people like me, who love trying new ideas, exploring different modalities, and working on real problems like biological sequences, Transformers come with a catch: **they're resource-hungry**. Running them at scale isn't always trivial, especially when you don't have access to large compute setups.

That’s why **Mamba** caught my eye.

Mamba is a relatively new architecture that promises something amazing: performance **comparable to Transformers**, but with **greater speed and efficiency**, especially on long sequences. It offers linear-time computation, which is a dream when you're working with massive biological data or need to iterate fast on experiments.

And yet... **nobody’s talking about it**. Not as much as Transformers, at least. Try to understand the Mamba architecture from scratch, and you’ll quickly realize how few intuitive resources are out there. That’s what motivated me to build this repo, to dive deep into how Mamba works, why it matters, and how it compares to the current state of the art.

We'll cover:
- What Mamba is and how it's built  
- Why it's a serious contender to Transformers  
- A hands-on example using **biological sequences** to help you understand how Mamba actually works in practice

> *P.S. - Sorry tech folks, but biology is the playground I love. If you’ve stumbled onto this repo, I hope the biological angle draws you in, not out!*

## 🔥 The Problem with Transformers

Transformers revolutionized sequence modeling by introducing **self-attention**, allowing each token to attend to every other token in a sequence. This kind of global interaction is incredibly powerful, but it comes at a steep cost.

Let’s say you’re working with **very long sequences**, like:

- A full mRNA transcript  
- A protein with hundreds of residues  
- A long DNA segment for epigenetic pattern prediction  
- Time-series data from clinical trials  

In all of these, you’re not just interested in local motifs, you often care about **long-range dependencies**. And here’s where Transformers begin to struggle:

- **Self-attention scales quadratically** with sequence length, both in time and memory.  
- This makes them hard to scale to sequences of length >10K, especially on modest hardware.  
- Training becomes slow, inference becomes bulky, and tuning becomes expensive.  

Even with tricks like attention sparsity, chunking, or memory compression, the core issue remains: **Transformers treat every token as equally relevant to every other token, and compute full pairwise interactions regardless of necessity.**

## 🐍 Enter Mamba: A Smarter Approach to Sequence Modeling

While Transformers look at every token in the sequence simultaneously and compute full pairwise interactions, **Mamba takes a fundamentally different path**, one that’s built for **efficiency**, **scalability**, and **long-context reasoning**.

Mamba is based on **state-space models (SSMs)**, a class of models that process sequences **sequentially** while maintaining a **latent state** that evolves over time. Here’s what that means in practice:

- It **walks through the sequence** one token at a time  
- It **maintains a hidden state** that captures relevant past information  
- As it encounters each new token, it **updates its internal state**, blending past memory with the new input  
- It does this with a learned mechanism that **selectively decides what to retain, amplify, or forget**  

Instead of treating all token relationships equally (like Transformers), Mamba **learns a smooth, continuous flow of information**, letting the model naturally emphasize what matters, and ignore what doesn't.

This leads to two major benefits:

- **Linear time and memory complexity** with respect to sequence length  
- **Long-context retention** without the quadratic overhead of self-attention  

You can think of it as a **continuous-time processor** that models how information evolves, rather than brute-force comparing every element to every other element.

### 🔧 1. State Space Models (SSMs): The Engine Behind Mamba

Imagine a box that holds a hidden state, like a moving summary of what it has seen so far.

Every time a new token comes in:

The state gets updated via a formula:

```math
x_{t+1} = A x_t + B u_t
```

And the model produces an output using:

```math
y_t = C x_t + D u_t
```

These **A, B, C, D matrices** are like gears that determine:

- How much of the past should carry forward  
- How the current input (`u_t`) affects the output (`y_t`)  

So far, this is **linear and efficient**, no pairwise comparisons like attention.

### 🧠 2. Making It Selective: Mamba’s Twist

Here’s where Mamba gets really powerful, and fundamentally different from traditional SSMs.

In a basic state-space model, the matrices **A**, **B**, **C**, and **D** are fixed. But Mamba adds something game-changing: it **dynamically changes its internal computation based on the current input**.

For every new token, Mamba essentially asks:

> “What kind of memory operation do I need *for this specific input*?”

This leads us to **Selective State Space Models**, the core innovation behind Mamba.

---

### 🌀 But What *Is* a Convolution Kernel in This Context?

In Mamba, the model doesn't just passively update a hidden state. Instead, it generates a **convolution kernel** on the fly, customized per token, to determine how to blend past information into the present.

Let’s break this down from first principles:

---

#### 🤖 If You’re Familiar with CNNs (Convolutional Neural Networks):

- A **convolution kernel** in CNNs is a small filter that slides over an image, detecting local patterns like edges or textures.
- It “blends” or “mixes” pixel values in a local window using learned weights.
- The key intuition: **You extract meaning from a local region using weighted combinations.**

---

#### 🧠 Now, in Mamba:

- Think of the input sequence as a **1D signal over time** (e.g., a long string of tokens, gene bases, or time-series values).
- Mamba creates a **custom convolution kernel** for *each* token, a vector of weights that tells the model:
  
  > "How much should each previous token influence the current output?"

- Instead of being fixed like in CNNs, these kernels are:
  - **Token-dependent** (they change per input)
  - **Learned dynamically** from the model itself
  - **Applied across time** (i.e., the sequence dimension)

---

### 🧮 And If You’re Thinking in Terms of Kernel Methods (like in SVMs):

- A **kernel** is a function that defines similarity or interaction between inputs.
- Mamba's kernels do something similar: they define how strongly past tokens interact with the current input, not via explicit comparison, but by **blending signals across time**.
- So you can think of Mamba’s kernel as a **temporal mixing function**.

---

### 🧪 Summary: How Mamba "Selects" What to Remember

So far, we had:

```math
x_{t+1} = A x_t + B u_t

```


Now in Mamba, it becomes more like:

```math
y_t = K_t * u_{0:t}

```


Where:

- `K_t` is a **learned kernel**, specific to the current token
- `u_{0:t}` is the history of inputs up to time `t`
- `*` denotes **convolution**, mixing inputs with the kernel

In other words:

> For each token, Mamba decides **how to filter the past** before incorporating it into the present, all without explicit attention.

---

#### 🔍 Attention vs. Mamba — A Comparison of Selection

| Transformer (Attention)                | Mamba (Selective SSM)                              |
|----------------------------------------|----------------------------------------------------|
| Computes pairwise similarity (dot-product) | Computes dynamic filters per token               |
| Asks: “Who should I attend to?”        | Asks: “How should I blend the past into now?”      |
| Quadratic in sequence length           | Linear in sequence length                          |
| Global communication at each step      | Local, learned memory blending over time           |

By combining the sequential nature of SSMs with input-conditioned convolution kernels, Mamba achieves something unique:  
**Efficient, continuous-time, token-aware memory**, perfect for long, structured sequences like those in biology.

---

### 🔄 3. Token Mixing, via a Fast Scan (Instead of Attention)

Once Mamba generates the dynamic convolution kernel for each token, it doesn’t just apply it in isolation.

Instead, it **scans over the sequence**, applying the kernel to blend each token's input with its local context, i.e., a **context-aware memory update**. This is called **token mixing**.

---

#### 🧠 What Does “Token Mixing” Mean?

In Transformers, token mixing happens via **self-attention**, every token looks at every other token and computes pairwise interactions.

In Mamba, token mixing happens through a **fast, learned convolutional scan**. At every step:

- The model **slides the kernel** over the sequence.
- Each token receives a **weighted blend of recent inputs**, where the weights are chosen *based on the current input itself* (as explained earlier).
- This blending is **efficient**, it runs in **linear time** and **constant memory** relative to sequence length.

You can think of this as applying a **custom lens over the sequence**, where each token decides:
> "Based on what I am, how should I interpret the past?"

This is fundamentally different from attention, which instead asks:
> "Which tokens should I look at, and how much?"

Because Mamba doesn’t compute full pairwise comparisons, it can scale easily to **sequences of millions of tokens**, which is a game-changer for domains like genomics and long medical time series.

---


### 🚪 4. Gate It Like an LSTM, Controlled Memory Flow

After token mixing, Mamba applies a **gating mechanism**, similar in spirit to what you might know from LSTMs or GRUs.

The final output is computed as:

```math
output = ssm_output * gate

```

Here’s the intuition:

- The **SSM output** is a filtered, memory-blended signal based on the convolutional scan.
- The **gate** is a learned, input-dependent value (between 0 and 1) that controls how much of that output should be *kept* or *suppressed*.

---

#### 🧠 Why Use a Gate? (And How Is It Like an LSTM?)

In LSTMs, gates help decide:
- What part of the memory should be forgotten
- What should be added to the new state
- What should be output

This is crucial for managing **long-term dependencies** and **non-linear transitions**.

Similarly, in Mamba:
- The **gate acts as a filter on memory flow**.
- It allows the model to **suppress irrelevant memory traces**, **enhance important signals**, or even **zero out the update** entirely.
- This adds a **non-linear decision layer** on top of the linear state-space scan, giving Mamba the expressive power of gated RNNs, but in a modern, efficient framework.

---

#### 🔧 Why This Matters

Without the gate, Mamba would be limited to *linear transformations* of past inputs — which may not be expressive enough for complex tasks.

The gate:
- Injects **non-linearity**
- Enables **dynamic control**
- Helps the model **decide how much memory to pass on**, just like an LSTM deciding how much of its cell state to retain

Finally, the gated output is passed through a **projection layer** — ensuring the output matches the expected model dimensionality (like the output of an MLP in a Transformer block).

By combining:
- **Dynamic kernels** (to filter memory),
- **Fast scans** (for efficient token mixing), and
- **Gates** (for non-linear control),

Mamba builds a **lightweight yet powerful memory system** — one that learns *how to remember*, *what to forget*, and *how to blend* context, all while staying fast and scalable.


---

## 🧬 Mamba Meets Biology: Learning from Codon Sequences

Let’s ground everything we've learned so far using a real biological example — an **mRNA codon sequence**.

---

### 📘 The Setup: A Codon Sequence

Consider a short mRNA snippet:

```python
AUG  UUU  GGC  CGA  UAA
```


Which translates to:

- **AUG** – Start codon (Methionine)  
- **UUU** – Phenylalanine  
- **GGC** – Glycine  
- **CGA** – Arginine  
- **UAA** – Stop codon  

Each codon — a triplet of nucleotides — is a **semantic unit**, or a “word,” in the language of biology.  
Even when codons map to the same amino acid (i.e., are **synonymous**), they can influence translation differently — affecting:
- Ribosomal speed  
- mRNA stability  
- Co-translational folding  

We want Mamba to learn these nuanced dependencies in sequence — and here’s how it does it.

---

### 🧠 Step-by-Step: Mamba's Perspective on This Sequence

---

#### 🧩 Step 1: Input Representation

Each codon is first embedded into a dense vector space:

```python
codon = ["AUG", "UUU", "GGC", "CGA", "UAA"]
embeddings = codon_embed(codon)
```

This results in a sequence of embeddings:

```math
E₀  E₁  E₂  E₃  E₄
```

Each Eᵢ encodes biologically relevant features such as:

- Codon frequency or bias
- GC content
- Amino acid identity
- mRNA secondary structure context
- Evolutionary conservation
- This forms the input to the Mamba model.

---

#### 🔁 Step 2: Sequential Scanning with Selective Memory

Unlike Transformers, which compute pairwise comparisons between all codons, Mamba processes the sequence step-by-step, using a dynamically filtered memory:

```python
x₀ = init_state
for each Eᵢ:
    kernelᵢ = f(Eᵢ)                # Learn token-specific convolution kernel
    xᵢ = A(Eᵢ) * xᵢ₋₁ + B(Eᵢ) * Eᵢ  # Update memory state
    yᵢ = C(Eᵢ) * xᵢ + D(Eᵢ) * Eᵢ   # Compute token-wise output
```

Here’s what’s happening:

- Each codon generates its own kernel — a custom temporal filter based on its properties.
- This kernel controls how the memory state updates, shaping how much the model remembers or forgets.
- The updated state xᵢ acts like a running biological context — much like how the ribosome experiences the transcript linearly.

---

#### 🚪 Step 3: Gating the Output

After computing yᵢ, Mamba applies a gate:

```python
outputᵢ = gate(Eᵢ) * yᵢ
```

This allows the model to selectively control which codons influence the final decision.

Why gating matters:

- Not every codon is equally informative in every context.
- Synonymous codons may only matter under specific structural or positional settings.
- The gate can suppress noise, enhance key signals, and inject non-linearity to improve modeling power.

This mirrors biological logic:

> "Should this codon influence the mRNA's predicted stability?"


## 🔬 Real Use Case: Predicting mRNA Stability
Imagine training the model to classify an mRNA as stable or degraded, based on codon usage.

Here's how Mamba might process our toy sequence:

- AUG → Start codon — likely triggers state initialization
- UUU → Rare codon — may introduce ribosome pausing
- GGC → Common codon — leads to smooth elongation
- CGA → Known to cause stalls — may increase dwell time
- UAA → Stop codon — affects polyadenylation, decay signals

As Mamba walks this sequence, it:

- Dynamically adjusts its internal state to reflect translation dynamics
- Learns when and how to gate token outputs based on context
- Builds a memory trace of biologically meaningful transitions

Eventually, this accumulated memory can be used for downstream tasks like:

- Predicting stability or degradation rates
- Modeling protein folding co-translationally
- Simulating ribosome dynamics


## 📄 Case Study: Mamba in "Orthrus — Towards Evolutionary and Functional RNA Foundation Models"

Now that we've built a strong intuition for how Mamba works, let’s look at how it performs in the real world.

In this section, we’ll briefly explore the paper **"Orthrus: Towards Evolutionary and Functional RNA Foundation Models"**, where the authors used Mamba to model RNA transcripts, and achieved **better performance than Transformer-based architectures**.

The paper makes a compelling case for using state-space models in biological sequence modeling, especially when it comes to capturing both **evolutionary signals** and **functional roles** encoded in RNA.

Let’s dive into how they integrated Mamba, what tasks they evaluated, and why Mamba outperformed the attention-heavy baselines.

### 🧬 What Is Orthrus?

Orthrus is a **foundation model trained on RNA sequences** with two key innovations:

1. **Uses Mamba** — not a Transformer, as the backbone encoder.  
2. **Trained via contrastive learning** — encouraging embeddings of functionally similar RNAs (like splice isoforms and orthologs) to cluster together.

---

### 💡 Why Is This Important?

RNA sequences present unique modeling challenges:

- **Long sequences**: Full-length mRNAs can exceed 12,000 nucleotides.  
- **Functional complexity**: Function is encoded not just in the protein product, but in:
  - RNA structure  
  - Splicing patterns  
  - Motif spacing  
  - Synonymous codon usage  
- **Small changes, big effects**: Even minor edits (e.g., synonymous substitutions or splice shifts) can affect stability, translation, or localization.

#### 🧱 The Problem with Existing Models

Most past models (e.g., DNA-BERT, Nucleotide Transformer) use **Masked Language Modeling (MLM)** — but:

- MLM forces the model to "guess" masked bases — even in **non-informative** regions (like intergenic or unselected sequences).
- MLM doesn't reflect **biological similarity** — such as isoforms with identical function but different sequences.

Orthrus addresses these issues by modeling **functional similarity**, not token recovery.

---

### 🐍 Why Orthrus Uses Mamba

Instead of Transformers, Orthrus uses Mamba, and for good reason:

| Feature                          | Why It Helps for RNA Modeling                                    |
|----------------------------------|-------------------------------------------------------------------|
| 🔁 Linear memory & time         | Handles long RNA sequences without truncation                    |
| 🧠 Sequential processing        | Naturally fits mRNA’s 5'→3' directional structure                 |
| 🎯 Selective memory            | Adapts dynamically to exon-intron boundaries, motifs, structure   |
| 🚫 No full attention needed    | Filters out irrelevant contexts, focuses on what matters         |

Mamba provides a **continuous, efficient, and biologically aware memory system** ideal for the complexity of RNA.

---

### 🎯 Training Objective: Biologically-Inspired Contrastive Learning

Instead of MLM, Orthrus uses a **contrastive learning approach** based on evolutionary and functional similarity.

#### ✅ Positive Pairs:
- Splice isoforms from the same gene  
- Orthologous transcripts across 400+ mammalian species (from the Zoonomia Project)

#### 🔧 Method:

1. For a given RNA, find a related isoform or ortholog.
2. Encode both with the **same Mamba encoder**.
3. Apply **Decoupled Contrastive Loss (DCL)**:
   - Pull functionally similar RNAs together.
   - Push unrelated sequences apart.

> 🧬 "These RNAs may differ in sequence — but do the same job. Their embeddings should be close."

---

### 📦 Input Representation: A Rich Biological Context

Orthrus doesn’t just use raw nucleotide sequences. It uses a **6-track input representation**, giving Mamba deep biological signals:

- ✅ Nucleotide bases (A, U, C, G)  
- 🔁 Codon start flags  
- 🔪 Splice site markers  
- 🔬 Structure indicators or masks  
- 🧬 Functional annotations (where available)

This allows the model to learn:
- Where translation starts  
- Where exon boundaries occur  
- How synonymous codons vary functionally  
- How structural elements affect expression

---

### 🧪 What Orthrus Learns: Downstream Tasks

After pretraining, Orthrus embeddings were evaluated on real RNA biology tasks:

| Task                               | Description                                      |
|------------------------------------|--------------------------------------------------|
| 🕒 RNA Half-Life                  | Predict mRNA stability                          |
| 🍽️ Ribosome Load                | Predict translation efficiency                  |
| 📍 Protein Localization           | Infer subcellular location                      |
| 🧠 GO Function                    | Predict biological process/function             |
| 🧬 Structural Properties          | Exon count, UTR length, etc.                    |

#### 📊 Results:

- Outperforms **DNA-BERT2, HyenaDNA, RNA-FM**
- Matches or beats **supervised models** like Saluki
- Strong in **few-shot learning** settings (e.g., <50 samples)

---

### 🧬 Bonus: Isoform-Level Understanding

Orthrus’s embeddings naturally cluster functionally similar isoforms — even when their sequences differ.

> Example: **BCL2L1 Gene**
- One splice isoform **promotes apoptosis**  
- Another **prevents apoptosis**  
- Orthrus embeddings cluster each group separately — reflecting **functional diversity** learned through contrastive training.

---

### ✨ Key Takeaways

| Concept              | Orthrus Approach                                   | Why It Matters for RNA |
|----------------------|----------------------------------------------------|-------------------------|
| **Architecture**     | Mamba-based state-space model                     | Scalable, structured memory |
| **Learning**         | Contrastive, not masked prediction                 | Captures biological similarity |
| **Input Encoding**   | 6-track (codons, splice sites, etc.)              | Context-aware learning |
| **Downstream Power** | Predicts stability, function, translation         | Learns codon-level regulation |
| **Strength**         | Few-shot + generalizable                          | Ideal where experiments are limited |

---

Orthrus showcases the true promise of Mamba:  
> A model that doesn’t just “process tokens,” but **understands sequences** in ways that reflect the underlying biology.



---

## 🧠 Final Thoughts

Mamba challenges the dominance of attention-based architectures with an elegant and efficient alternative, one that doesn't just scale better, but **thinks differently**.

From abstract sequence modeling to **real-world biological data like RNA**, Mamba shows its strength in tasks that demand both **long-range understanding** and **local, context-sensitive precision**.

Whether you're a systems researcher, a bioinformatician, or someone fascinated by the intersection of AI and biology, Mamba is a model worth watching, experimenting with, and most importantly, understanding.

🔧 For those interested in experimenting with Mamba, this [repository](https://github.com/alxndrTL/mamba.py.git) by Alexandre Torres–Leguet offers an easy-to-follow PyTorch implementation. Definitely worth a look!

---

## 📚 References

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
   Albert Gu, Tri Dao, et al. (2023)  
   [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

2. **Orthrus: Towards Evolutionary and Functional RNA Foundation Models**  
   Fradkin, P., Shi, R., Isaev, K., Frey, B.J., Morris, Q., Lee, L.J. and Wang, B., 2024 
   [https://www.biorxiv.org/content/10.1101/2024.03.11.583046v1](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v1)

3. **The genetic and biochemical determinants of mRNA degradation rates in mammals**  
   Agarwal, V., & Kelley, D. R. (2022). 
   [https://pubmed.ncbi.nlm.nih.gov/36419176](https://pubmed.ncbi.nlm.nih.gov/36419176/)

4. **Hyenadna: Long-range genomic sequence modeling at single nucleotide resolution**
   Nguyen, E., Poli, M., Faizi, M., Thomas, A., Wornow, M., Birch-Sykes, C., ... & Baccus, S. (2023).
   [https://proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html)

5. **Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions**
   Chen, J., Hu, Z., Sun, S., Tan, Q., Wang, Y., Yu, Q., ... & Li, Y. (2022)
   [https://arxiv.org/abs/2204.00300](https://arxiv.org/abs/2204.00300)

6. **Alexandre Torres–Leguet.**
   *mamba.py: A simple, hackable and efficient Mamba implementation in pure PyTorch and MLX.*
   Version 1.0, 2024. Available at: [https://github.com/alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py)
---

Thanks for reading — and stay curious! 🧬🐍





