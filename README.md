# Intel Unnati Industrial Training 2024 - Disease Symptom Prediction Chatbot Using Llama-2-7b through Intel Extension for Transformers
PS04: Introduction to GenAI and Simple LLM Inference on CPU and finetuning of LLM Model to create a Custom Chatbot


## Problem Statement

Despite advancements in medical technology, individuals often lack accessible, reliable, and timely information regarding potential health issues and associated precautions. Many delay seeking medical advice due to an inability to recognize early symptoms or understand their significance, leading to the worsening of conditions that could have been managed or mitigated with early intervention.

## Solution

We propose a symptom-based disease prediction chatbot powered by the Llama-2 model, fine-tuned using Intel® OpenVINO™ for efficient performance on Intel AI laptops. This chatbot assists users by interpreting their symptoms and predicting potential diseases, offering timely and reliable medical advice.

### Features

- **Interactive Chatbot Interface**: Allows users to input their symptoms and receive detailed responses.
- **Symptom-Based Disease Prediction**: Utilizes a custom dataset for accurate mapping of symptoms to diseases.
- **Machine Learning Model**: Leverages a fine-tuned Llama-2 model for precise predictions.
- **Real-Time Response**: Provides immediate recommendations based on user inputs.
- **Scalable and Flexible**: Designed for future updates and enhancements.

## Functional Process

### Setup

1. **Install Required Packages**:
   - Install essential Python packages: `accelerate`, `peft`, `bitsandbytes`, `transformers`, and `trl`.

2. **Define Variables**:
   - Set key variables like the model name (`NousResearch/Llama-2-7b-chat-hf`), dataset name (`vl8222/Disease_Symptom_Prediction_v5-1k`), and fine-tuned model name (`Llama-2-7b-chat-finetune-for-disease-prediction`).

### Hugging Face Authorization

1. **Login to Hugging Face Hub**:
   - Authenticate using `!huggingface-cli login` to upload the fine-tuned model to the Hugging Face Hub.

2. **Load the Model and Tokenizer**:
   - Use `AutoModelForCausalLM` and `AutoTokenizer` from the `transformers` library to load the pre-trained Llama-2 model and tokenizer.

### Model Fine-Tuning and Optimization

1. **Load Dataset and Prepare for Training**:
   - Use `load_dataset` to load and prepare the dataset for training.

2. **Fine-Tune the Model**:
   - Employ `SFTTrainer` to fine-tune the model with the dataset, setting training parameters like epochs, batch size, and learning rate.

3. **Configure BitsAndBytes**:
   - Implement 4-bit quantization using `BitsAndBytesConfig` to optimize memory usage and efficiency.

4. **Apply Low-Rank Adaptation (LoRA)**:
   - Use `LoraConfig` to apply LoRA for parameter-efficient fine-tuning.

### Model Inference

1. **Generate Predictions**:
   - Create a text generation pipeline to interpret input symptoms and generate disease predictions.

## Tech Stack

- **Core Components**:
  - **Large Language Model**: Fine-tuned Llama-2.
  - **Framework**: Intel Extension for Transformers, Hugging Face Transformers.

- **Programming Language**: Python.

- **Core Modules**:
  - `os`, `torch`, `datasets`, `transformers`, `logging`, `peft`, `trl`, `intel_extension_for_transformers`.

## Demonstration


## Conclusion

![Screenshot 2024-07-08 222912](https://github.com/vlen4114/Intel-Unnati-Industrial-Training-2024/assets/113226055/d2ab0c0d-0828-45ce-b5df-2d786b2d8f25)

In this project, we fine-tuned the Llama-2 model for disease symptom prediction, employing techniques such as Low-Rank Adaptation (LoRA) and 4-bit quantization via the BitsAndBytes library for optimized performance. The fine-tuned model, hosted on the Hugging Face Hub, demonstrates the potential of transformer-based models in the healthcare domain, providing a robust foundation for developing advanced tools for disease diagnosis and management.


---

**Note**: This project leverages the capabilities of Intel® OpenVINO™ and transformer-based models to deliver an efficient and scalable solution for symptom-based disease prediction.


