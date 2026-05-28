# VictorianGPT: A Compound AI System

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/tm-vettel/VictorianGPT)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)]()

VictorianGPT is an end-to-end Machine Learning pipeline that combines **Parameter-Efficient Fine-Tuning (QLoRA)** with **Retrieval-Augmented Generation (RAG)** to create a highly atmospheric, 19th-century conversational agent. 

The system acts as a sophisticated, brooding scholar, grounding its philosophical responses in the authentic texts of Charles Dickens, Bram Stoker, Emily Brontë, and Oscar Wilde.

## 🚀 Live Demo
**Try the deployed application here:** [VictorianGPT on Hugging Face](https://huggingface.co/spaces/tm-vettel/VictorianGPT)

<div align="center">
  <img src="assets/app.png" alt="VictorianGPT Web Interface" width="850"/>
</div>
---

## 🧠 System Architecture
This project is built as a Compound AI System, moving from raw unstructured data to a cloud-hosted inference endpoint.

1. **Data Pipeline:** Extracted, cleaned, and chunked raw textual data from Gutenberg 19th-century literature. Generated ~10,000 synthetic conversational pairs for aesthetic alignment.
2. **Fine-Tuning (The Voice):** Fine-tuned `Qwen/Qwen2.5-3B-Instruct` using QLoRA (4-bit quantization) to align the model's tone with Dark Academia and Victorian gothic prose.
3. **Semantic Memory (The Brain):** Built a local Vector Database using `ChromaDB` and `sentence-transformers/all-MiniLM-L6-v2`. The RAG engine injects actual historical paragraphs into the context window at inference time.
4. **Orchestration & Deployment:** Wrapped the model and retrieval engine in a unified generation loop, heavily penalizing repetition (`repetition_penalty=1.15`) to prevent context domination, and deployed via Gradio to Hugging Face Spaces.

## 🛠️ Tech Stack
* **LLM Foundation:** Qwen 2.5 (3B Parameters)
* **Fine-Tuning:** PyTorch, Hugging Face `transformers`, `peft` (LoRA), `bitsandbytes` (NF4 Quantization)
* **RAG Pipeline:** LangChain, ChromaDB, Sentence Transformers
* **Deployment:** Gradio, Hugging Face Spaces

---

## 📂 Repository Structure
* `/notebooks`: Contains the Jupyter notebooks detailing the data pipeline, the PyTorch tensor routing for QLoRA, and the embedding generation for ChromaDB.
* `/src`: Contains `app.py`, the production inference script running the live RAG orchestration.

## ⚙️ Local Installation
If you wish to run the inference engine locally, you will need to download the fine-tuned adapter and the Chroma database from the Hugging Face Space repository.

```bash
git clone [https://github.com/](https://github.com/)[YOUR-GITHUB-USERNAME]/VictorianGPT.git
cd VictorianGPT
pip install -r requirements.txt
python src/app.py
