---
title: Large Language Models
teaching: 45
exercises: 20
---

::::::::::::::::::::::::::::::::::::::: objectives

- Define both a trivial reference baseline and a practical model
  basket.
- Choose an initial model based on task type, data shape,
  interpretability, and time available.
- Distinguish between a first baseline model and a stronger comparison
  model.
  
::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- What counts as a sensible baseline or comparison model?
- Which conventional models belong in my starter model basket?

::::::::::::::::::::::::::::::::::::::::::::::::::

1. What is a Language Model?

A language model learns the statistical structure of language by estimating the probability of a sequence of tokens (words or subwords).

Formally, it models:

P(w1,w2,...,wn)
P(w
1
	​

,w
2
	​

,...,w
n
	​

)

Using the chain rule of probability, this can be decomposed as:

P(w1,...,wn)=∏t=1nP(wt∣w1,...,wt−1)
P(w
1
	​

,...,w
n
	​

)=
t=1
∏
n
	​

P(w
t
	​

∣w
1
	​

,...,w
t−1
	​

)

This means the model predicts each token based on all previous tokens.

Example:
Input: “The capital of France is”
Prediction: “Paris”

By repeatedly predicting the next token, the model can generate full sentences and paragraphs.

2. Tokens and Tokenization

Large language models do not process raw text directly. Instead, text is converted into tokens—numerical representations of words or subword units.

Example sentence:
“Machine learning models are powerful.”

Possible tokenization:
["Machine", "learning", "models", "are", "powerful", "."]

Common tokenization methods include:

Byte Pair Encoding (BPE): Splits words into frequent subword units
WordPiece: Widely used in transformer-based models
SentencePiece: Language-independent and works directly on raw text

Tokenization enables models to efficiently handle large and diverse vocabularies.

3. Architecture of Large Language Models

Most modern LLMs are based on decoder-only versions of the Transformer architecture.

A typical pipeline consists of:

Token embeddings
Positional encodings
Stacked transformer layers
Output probability distribution

Simplified flow:

Input Tokens
↓
Embedding Layer
↓
Transformer Blocks (Self-Attention + Feedforward)
↓
Linear Projection
↓
Softmax
↓
Next Token Probabilities

Each transformer layer uses self-attention to allow tokens to attend to earlier tokens in the sequence, enabling rich contextual understanding.

4. Training Large Language Models

Training a large language model (LLM) typically occurs in multiple stages, each improving the model’s capabilities.

Pretraining

The first stage is self-supervised learning on massive text datasets.

Objective:
Predict missing or next tokens in a sequence.

Common training tasks:

Next-token prediction: Predict the next word in a sequence
Masked language modeling: Predict missing words within a sentence
Sequence completion: Continue a given passage

Typical training data includes:

Books
Web pages
Scientific articles
Code repositories
News archives

This stage provides the model with broad knowledge of language, grammar, and general world concepts.

Fine-Tuning

After pretraining, the model is adapted to specific tasks using smaller, labeled datasets.

Examples:

Question answering
Summarization
Dialogue generation
Code completion

Fine-tuning improves performance on targeted applications.

Instruction Tuning

Instruction tuning trains the model to better follow human instructions.

Example:

Instruction: “Explain photosynthesis in simple terms.”
Response: A clear, simplified explanation

This stage improves:

Usability
Alignment with user intent
Task-specific performance
Reinforcement Learning from Human Feedback (RLHF)

Many LLMs are further refined using Reinforcement Learning from Human Feedback.

Process:

Humans rank multiple model outputs
A reward model learns these preferences
Reinforcement learning updates the model to favor better responses

Outcomes:

More helpful answers
Safer and more aligned outputs
Improved reasoning and coherence

![](fig/LLMS_summary.png){alt="Diagram showing a one-dimensional convolution scanning across a sequence."}


## Key points

:::::::::::::::::::::::::::::::::::::::: keypoints
- Choose the task type before choosing the algorithm.
- A good starter model basket includes both simple baselines and one or
  two stronger comparison options.
- Conventional models are usually the right first step for structured
  or limited data.
- Stronger models should be added for a reason, not because they sound
  more advanced.
::::::::::::::::::::::::::::::::::::::::::::::::::
