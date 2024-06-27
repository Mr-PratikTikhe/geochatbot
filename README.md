# GeoChatBot and Anomaly Map Generator

This is the GeoChatBot, created for the hackathon organized by Mineral Exploration and Consultancy Limited (MECL). The project includes a Streamlit application that features two main functionalities:

1. **GeoChatBot**: An interactive chatbot to answer questions related to geological documents.
2. **Anomaly Map Generator**: A tool to generate interpolated anomaly maps using IDW (Inverse Distance Weighting) and Kriging methods for various elements.

## Features

### GeoChatBot 🤖

- Utilizes [LangChain](https://github.com/langchain/langchain) , [LLaMA 2 7B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) and CTransformers for natural language processing and document retrieval.
- Provides answers based on geological documents in PDF format.
- Maintains a conversation history for better context understanding.

![GeoChatBot Screenshot] ![Screenshot 2024-06-27 192245](https://github.com/Mr-PratikTikhe/geochatbot/assets/142296701/4400d250-e3eb-4644-9175-b8dd9551ee2a)
![Screenshot 2024-06-27 193045](https://github.com/Mr-PratikTikhe/geochatbot/assets/142296701/fc7885fd-7f71-4754-8852-477867e3b9c9)



### Anomaly Map Generator 🗺️

- Generates interpolated maps for geological elements using IDW and Kriging methods.
- Allows users to zoom, pan, and draw on the maps to explore anomalies in detail.
- Measures distance between selected points on the map.

![Anomaly Map Generator Screenshot]![Screenshot 2024-06-27 193118](https://github.com/Mr-PratikTikhe/geochatbot/assets/142296701/0091825a-d1a2-4b53-9d0a-e922775f8af8)

- By IDW method.
  ![Screenshot 2024-06-27 193148](https://github.com/Mr-PratikTikhe/geochatbot/assets/142296701/9c69548b-bdef-4ab9-9331-b221b5bba63a)

- By Kriging method.
  ![Screenshot 2024-06-27 193205](https://github.com/Mr-PratikTikhe/geochatbot/assets/142296701/6f8aad21-74ba-4d17-a0c3-a02931db13b4)

- Loats of features are there.
  ![Screenshot 2024-06-27 193315](https://github.com/Mr-PratikTikhe/geochatbot/assets/142296701/7e00aa6e-97ee-42cd-8d2b-7897bb1c61ac) 
  



## Installation

To run this application locally, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Create a virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your data:**

   - Place your geological PDF documents in the `data/` directory.
   - Ensure you have a CSV file with your data for generating anomaly maps (e.g., `Analytical_value_55K03.csv`).

2. **Run the Streamlit application:**

    ```sh
    streamlit run mian.py
    ```

3. **Navigate the application:**

   - Use the sidebar to switch between the GeoChatBot and the Anomaly Map Generator.
   - For the GeoChatBot, type your questions and get answers based on the loaded documents.
   - For the Anomaly Map Generator, enter the name of the element you want to analyze and explore the generated maps.

## Dependencies

The required dependencies for this project are listed in the `requirements.txt` file:

```txt
streamlit==1.23.1
streamlit-chat==0.0.2
langchain==0.0.138
langchain_community==0.0.12
numpy==1.23.3
pandas==1.5.2
folium==0.13.0
scikit-learn==1.1.3
scipy==1.9.3
streamlit-folium==0.10.0
branca==0.5.0
matplotlib==3.6.2
base64==1.0.0
