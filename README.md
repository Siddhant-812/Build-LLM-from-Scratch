# Build an LLM From Scratch: A Complete Showcase

This project is a comprehensive, hands-on implementation of a GPT-style Large Language Model (LLM) built entirely from scratch using Python and PyTorch. The journey takes us from the fundamental building blocks of data tokenization to pretraining a base model, and finally to advanced, memory-efficient finetuning techniques like LoRA and QLoRA.

The project culminates in an interactive **Streamlit application** that showcases the capabilities of every model we built and provides a visual analysis of their performance and efficiency.

## 🚀 Features

  * **Modular Architecture**: Every component, from the tokenizer to the Transformer blocks, is built in a clean, modular, and easy-to-understand way.
  * **Pretrained Base Model**: A generative model trained on the TinyShakespeare dataset that can produce text in a Shakespearean style.
  * **Sentiment Analysis**: The base model is finetuned for sentiment classification using two different methods:
      * **Full Finetuning**: All model weights are updated.
      * **LoRA Finetuning**: A parameter-efficient method that trains less than 1% of the model's weights.
  * **Instruction Following**: The base model is finetuned using the cutting-edge **QLoRA** technique on the Dolly-15k dataset, teaching it to follow user instructions.
  * **Interactive Showcase**: A full-featured Streamlit web application to interact with every model and visualize performance metrics.
  * **Quantitative Evaluation**: A script to measure the base model's performance using the **perplexity** metric on an unseen test set.
    
## Interface

<img width="1832" height="727" alt="llm1" src="https://github.com/user-attachments/assets/214477b1-5c73-4b70-9813-0abf1768d2fe" />
<img width="1880" height="782" alt="llm2" src="https://github.com/user-attachments/assets/8ae89e66-df51-4c79-b6ba-02251d627c91" />
<img width="1871" height="912" alt="llm3" src="https://github.com/user-attachments/assets/4d26d4c5-2d35-4b16-ad54-74462a96c438" />
<img width="1781" height="521" alt="llm4" src="https://github.com/user-attachments/assets/4a97de9e-d4f2-458c-91f3-3622f63d5794" />
<img width="1587" height="627" alt="llm5" src="https://github.com/user-attachments/assets/f1c42d6d-d319-4998-a7f6-f84274097758" />

## 📂 Project Structure

Here is a breakdown of what each script in the project does:

| File Name | Purpose |
| :--- | :--- |
| `main.py` | The main script for pretraining the base GPT model on the TinyShakespeare dataset and saving its weights. |
| `model.py` | Defines the core neural network architecture, including the `SelfAttention`, `MultiHeadAttention`, `TransformerBlock`, and `GPTModel`. |
| `tokenizer.py` | Contains the `BPETokenizer` class, which handles text-to-token conversion using Byte-Pair Encoding. |
| `data_utils.py` | A utility script with the `create_dataloader` function to efficiently prepare and serve data in batches for training. |
| `training.py` | Contains the main `train_model` function, which orchestrates the entire training loop: forward pass, loss calculation, backpropagation, and weight updates. |
| `generation.py` | Includes the `generate_text` function to produce new text from a model, featuring advanced sampling methods like temperature scaling and top-k. |
| `evaluate.py` | A script to quantitatively measure the performance of the pretrained model on a test dataset using the perplexity metric. |
| `lora.py` | Defines the `LoRALayer` class, a custom module that implements the Low-Rank Adaptation technique. |
| `finetune_classifier.py` | A script for **full finetuning** of the base model for a sentiment analysis task. |
| `finetune_lora_classifier.py`| A script demonstrating parameter-efficient finetuning for sentiment analysis using **LoRA**. |
| `finetune_qlora_instructions.py`| The most advanced script, which finetunes the model for instruction-following using **QLoRA** (4-bit Quantization + LoRA). |
| `app.py` | The final Streamlit web application that provides an interactive UI to test all the trained models and visualize results. |

## ⚙️ Setup and Usage

### 1\. Installation

First, clone the repository and install the required Python libraries. It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

*(You will need to create a `requirements.txt` file containing `torch`, `requests`, `streamlit`, `pandas`, `datasets`, and `bitsandbytes`)*

### 2\. Pretraining the Base Model

The first step is to pretrain our base GPT model. This will create the `model_weights.pth` file which is required for all other steps.

```bash
python main.py
```

This process will take a few minutes, especially if you are not using a GPU.

### 3\. Running the Finetuning Scripts

After pretraining, you can run any of the finetuning scripts. Each script uses the `model_weights.pth` file as a starting point.

```bash
# To run full classification finetuning
python finetune_classifier.py

# To run parameter-efficient LoRA finetuning
python finetune_lora_classifier.py

# To run advanced QLoRA instruction finetuning
python finetune_qlora_instructions.py
```

### 4\. Launching the Showcase Application

To see all the models in action, run the Streamlit application.

```bash
streamlit run app.py
```

This will open the interactive showcase in your web browser.

## 🗺️ Our Implementation Journey

We built this project in a phased, step-by-step manner, ensuring each component was understood before moving to the next.

### Phase 1: Data Preparation

We started by creating a robust **BPE Tokenizer** from scratch. This involved building a vocabulary by iteratively merging the most frequent pairs of tokens, allowing our model to understand sub-word units efficiently.

### Phase 2: Building the GPT Model

We constructed the GPT model layer by layer in `model.py`:

1.  **Embedding Layer**: Created token and positional embeddings to give the model a sense of word meaning and order.
2.  **Self-Attention**: Implemented a single, causal self-attention head, the core mechanism of a Transformer.
3.  **Multi-Head Attention**: Combined multiple attention heads to allow the model to capture richer contextual information in parallel.
4.  **Transformer Block**: Assembled the attention mechanism and a feed-forward network into a single, stackable block with residual connections and layer normalization.
5.  **Full GPT Model**: Stacked the Transformer blocks and added a final output layer to create the complete generative model.

### Phase 3: Pretraining

We implemented a training loop in `training.py` and a data loader in `data_utils.py`. We trained our model on the TinyShakespeare dataset to predict the next token in a sequence, saving the learned weights to `model_weights.pth`.

### Phase 4: Advanced Finetuning

This was the most advanced phase, where we adapted our pretrained model for specialized tasks:

1.  **Classification Finetuning**: We added a classification head and fully finetuned the model on a sentiment analysis dataset.
2.  **LoRA**: We implemented the LoRA technique to perform the same classification task by only training a tiny fraction of the parameters, showcasing a massive gain in efficiency.
3.  **QLoRA for Instructions**: We combined 4-bit quantization with LoRA to finetune the model on an instruction-following dataset, demonstrating how to customize large models on consumer hardware.

## 🚧 Challenges and Solutions

Throughout this project, we encountered and solved several real-world machine learning challenges:

  * **CUDA Device Mismatches**: We encountered an error where weights saved on one GPU device could not be loaded on another.

      * **Solution**: We made our loading process more robust by always mapping the loaded weights to the CPU first (`map_location='cpu'`) before moving the model to the target device. This is a best practice for portable PyTorch code.

  * **Tokenizer Vocabulary Mismatch**: Our model, pretrained on Shakespeare, crashed when it saw modern words during finetuning because the new tokens were outside its vocabulary range.

      * **Solution**: We implemented a **data sanitization** step. We first trained our tokenizer on a combined corpus of both datasets and then added a validation loop to discard any finetuning samples that still produced out-of-vocabulary tokens, guaranteeing the model would never see an invalid ID.

  * **Streamlit Caching Bugs**: The interactive nature of Streamlit caused our model modification code (for LoRA and quantization) to run multiple times, leading to errors.

      * **Solution**: We made our code idempotent by adding checks (`isinstance(layer, torch.nn.Linear)`) to ensure that we only wrap or quantize a layer if it hasn't already been modified.

  * **Gibberish Output**: Our instruction-finetuned model produced nonsensical answers.

      * **Solution**: This was not a bug, but a crucial lesson. We identified that the poor quality was an expected outcome due to the microscopic **scale** of our model, vocabulary, and datasets compared to state-of-the-art LLMs. This highlighted that the principles of building an LLM are accessible, but achieving high performance is a matter of massive scale.
