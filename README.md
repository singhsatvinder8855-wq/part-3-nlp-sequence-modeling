# part-3-nlp-sequence-modeling

# Part 3: Customer Support Ticket Text Classification & Sequence Modeling

### Dataset Source & Path
* **Original Dataset Link:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing
* **Project Folder File Path:** `ai_project_synthetic_datasets/part_3_nlp_sequence_modeling/customer_support_text_classification.csv`

---

## 1. Approach & Steps

* **Dataset Exploration (Task 1):** Loaded and inspected the customer support dataset containing ticket metadata and raw query text. Analyzed total dimensions, feature data types, calculated average message word counts, and checked target category distributions.
* **Text Preprocessing & Cleaning (Task 2):** Applied standard lowercasing across the text corpus and used regular expressions (`re`) to clean out unnecessary punctuation, symbols, and numerical noise. Standardized text into an engineered column `cleaned_text`.
* **Target Label Encoding:** Converted categorical text labels (`positive`, `neutral`, `negative`) into explicit numerical class values. Partitioned the processed dataset into 80% Training and 20% Testing subsets using stratified splitting to preserve class balances.
* **Feature Vectorization & Baseline Model (Tasks 3 & 4):** Extracted features from the textual data using a mathematical `TfidfVectorizer` capped at 5,000 maximum features. Configured and evaluated a baseline Multinomial Naive Bayes classification pipeline.
* **Advanced Sequence Modeling (Task 5):** Tokenized raw word sequences and applied sequence padding (`pad_sequences`) up to a uniform length of 150 tokens. Built a Deep Learning sequential network consisting of an `Embedding` layer, a recurrent `LSTM` layer with dropout normalization, and a multi-class `Dense` output layer configured with a `softmax` activation function.

---

## 2. Results & Evaluation Outputs

All training logs, accuracy performance scores, and visual validation artifacts are strictly organized under the auto-generated `results/` folder layout:
* **Confusion Matrix Visualization:** `results/model_evaluation.png` (Graphical heatmap displaying the model's true classifications versus predicted classifications across all sentiment targets).
* **Sample Inference Logs:** `results/sample_predictions.txt` (Text log containing full un-truncated customer text examples along with their corresponding Ground Truth labels and Deep Learning model predictions).

---

## 3. Task 6: Attention and Transformer Reflection

### A. Why RNNs struggle with long-term dependencies
Standard Recurrent Neural Networks (RNNs) process text tokens sequentially, in a rigid word-by-word manner. When processing exceptionally long customer support messages or multi-paragraph tickets, error signals have to pass through a vast sequence of temporal steps during the backpropagation phase. 

This causes the gradients to shrink exponentially, resulting in the **Vanishing Gradient Problem**. Mathematically, the network's weights cease to update efficiently for early-stage tokens. Consequently, by the time the RNN reaches the final word of a long ticket, it completely loses track of critical context signals or tokens that were introduced at the very beginning of the string.

### B. How LSTMs help with memory
Long Short-Term Memory networks (LSTMs) bypass the vanishing gradient bottleneck by embedding a dedicated structural component called the **Cell State**. The cell state functions as an uninterrupted internal information highway running straight through the sequential layer chain, allowing information to pass with minimal modification. This highway is closely managed by three specialized mathematical "gates":
* **Forget Gate:** Evaluates historical context inputs and dynamically filters out irrelevant or redundant information to discard from memory.
* **Input Gate:** Decides which incoming raw text vectors or new contextual units are meaningful enough to be written into the cell state memory.
* **Output Gate:** Determines what the final hidden state output vector should look like, utilizing the newly updated, stable cell state information.

These specialized operations allow gradients to stay stable over time, enabling the neural network to successfully maintain deep, long-term semantic dependencies across extended chunks of text.

### C. What attention solves in sequence-to-sequence tasks
In old-school encoder-decoder systems, the encoder network was forced to compress a raw sequence of variable text lengths into a single, static, fixed-size context vector before passing it over to the decoder. This architecture created a severe information bottleneck; complex or highly detailed multi-sentence tickets could not fit perfectly inside a single numeric vector, leading to data loss.

**Attention mechanisms** completely eliminate this structural limitation. Instead of relying on a single final compressed state, attention gives the decoder direct access to *all* historical hidden states of the input text simultaneously. It computes dynamic mathematical weights at run time to "pay attention" specifically to the key phrase vectors in the query text that matter most for predicting the exact target class.

### D. Why transformers are important in modern NLP and Generative AI
Transformers completely revolutionized the modern Artificial Intelligence landscape by completely discarding recurrent sequential layer constraints (RNNs/LSTMs) and replacing them entirely with a framework called **Self-Attention**. This architectural shift yielded two monumental benefits:
* **True Parallel Processing:** Because sentences are no longer processed token-by-token in a rigid linear timeline, entire enormous text corpora can be fed into graphic processors (GPUs) simultaneously. This dramatically cuts down network training times from weeks to hours.
* **Global Context Awareness:** In a Transformer layer, every word calculates its semantic relationship with every other word in the text instantly, regardless of their distance or positional gap in a document. 

This parallel workflow and unconstrained context window provide the fundamental base for training massive, modern Large Language Models (LLMs like GPT-4 and Gemini), giving them the ability to generate and analyze highly complex human context flawlessly.
