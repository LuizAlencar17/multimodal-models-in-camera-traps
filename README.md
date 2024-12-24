# 📚 Multimodal Models in Camera Traps

---

## 🚀 Project Overview
**Multimodal Models in Camera Traps** is a comprehensive toolkit designed to process and analyze multimodal datasets (images and text) for tasks such as behavior classification and object recognition in wildlife camera trap datasets. This project utilizes state-of-the-art machine learning models like **CLIP**, **BLIP**, **GPT**, and **Gemini** for robust evaluation, training, and inference.

---

## 🛠️ Project Structure

The repository consists of the following key components:

- **`preprocess.py`**  
   - Prepares and processes image and text datasets for training and evaluation.
   - Includes specialized datasets:  
     - `TextImageDataset`  
     - `QuestionAnsweringDataset`  
     - `SimilarityDataset`

- **`utils.py`**  
   - Provides utility functions for image resizing, model retrieval, and label processing.  
   - Functions like `resize_image`, `get_model_and_processor`, and `get_prompt` streamline workflows.

- **`flags.py`**  
   - Central configuration file for hyperparameters and file paths (e.g., batch size, epochs, dataset paths).

- **`blip.py`**  
   - Implements training and evaluation workflows using **BLIP** (`Salesforce/blip-vqa-base`) for image and text tasks.

- **`clip.py`**  
   - Implements training and evaluation workflows using **CLIP** (`openai/clip-vit-base-patch32`) for visual similarity tasks.

- **`gemini.py`**  
   - Integrates **Google Gemini** for inference tasks.  
   - API integration supports image and text-based predictions.

- **`run.sh`**  
   - Shell script for executing different configurations for training and evaluation across tasks (`behavior`, `animal`) and models (`CLIP`, `BLIP`, `GPT`, `Gemini`).

- **`create_dataset.ipynb`** *(Optional)*  
   - Jupyter Notebook for dataset creation and preprocessing.

---

## ⚙️ Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/multimodal-models-in-camera-traps.git
   cd multimodal-models-in-camera-traps

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
3. **Environment Variables**
   ```bash
   Update API keys and dataset paths in the `flags.py` file.

# 📊 Usage
Training and Evaluation

1. You can execute the pipeline using the run.sh script:
   ```bash
    bash run.sh

2. Or run specific model tasks:

- For CLIP Model
   ```bash
    python3 clip.py --flagfile=configs/clip/clip_train_behaviour_serengeti_seq.config

- For BLIP Model
   ```bash
    python3 blip.py --flagfile=configs/blip/blip_train_behaviour_serengeti_seq.config

- For Gemini Model
    ```bash
    python3 gemini.py --flagfile=configs/gemini/gemini_test_behaviour_serengeti_seq.config

# 📑 Configuration
Adjust settings in flags.py:

- Dataset paths: train_filename, val_filename, test_filename
- Hyperparameters: batch_size, num_epochs, learning_rate
- Model Settings: task, model_name, tags

# 📈 Results
- Evaluation results are saved in:

   ```bash
    /data/luiz/dataset/results/behaviour-classifier/

# 🧠 Key Features
- Multimodal Integration: Supports image and text inputs for advanced analysis.
- Pre-trained Models: Seamless integration of models like CLIP, BLIP, Gemini, and GPT.
- Scalability: Configurable settings for batch sizes, learning rates, and dataset partitions.

# 🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository.
- Create a feature branch:
   ```bash
    git checkout -b feature-name

- Commit changes:
   ```bash
    git commit -m "Add new feature"

- Push to branch:
   ```bash
    git push origin feature-name

- Submit a Pull Request.
# 📝 License
This project is licensed under the MIT License.

# 📧 Contact
For questions or feedback, reach out to:

- Your Name: fabio.alencar644@gmail.com
- GitHub: LuizAlencar17