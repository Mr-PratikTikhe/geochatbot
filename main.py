import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np
import folium
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import Rbf
from folium.plugins import MeasureControl, LocateControl, Draw
from streamlit_folium import folium_static
import branca.colormap as cm
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from math import radians, sin, cos, sqrt, atan2

# Function to calculate distance between two points using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of Earth in km
    return distance

# Function to create an image from the interpolated values and return it as a base64 string
def create_interpolated_image_base64(interpolated_values, colormap):
    norm = plt.Normalize(vmin=interpolated_values.min(), vmax=interpolated_values.max())
    image = colormap(norm(interpolated_values))
    image = (image[:, :, :3] * 255).astype(np.uint8)
    
    buffer = BytesIO()
    plt.imsave(buffer, image, format='png')
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    
    return img_base64

# Function to create IDW interpolated map
@st.cache_data
def create_idw_interpolated_map(df, element):
    element_data = df[['longitude', 'latitude', element]].dropna()
    X = element_data[['longitude', 'latitude']].values
    y = element_data[element].values
    idw = KNeighborsRegressor(n_neighbors=4, weights='distance')
    idw.fit(X, y)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  # Reduce grid resolution
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    interpolated_values = idw.predict(grid_points).reshape(xx.shape)

    # Normalize values to percentage
    y_min_value, y_max_value = y.min(), y.max()
    interpolated_values_percent = 100 * (interpolated_values - y_min_value) / (y_max_value - y_min_value)

    # Create folium map
    m = folium.Map(location=[X[:, 1].mean(), X[:, 0].mean()], zoom_start=10)

    # Define colormap using original min and max values
    colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'red'], vmin=y_min_value, vmax=y_max_value, caption=element)

    # Create the interpolated image base64
    img_base64 = create_interpolated_image_base64(interpolated_values_percent, plt.get_cmap('plasma'))

    # Add interpolated values as a raster layer
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_base64}',
        bounds=[[y_min, x_min], [y_max, x_max]],
        opacity=0.7
    ).add_to(m)

    # Add original data points with percentage values
    for lat, lon, val in zip(X[:, 1], X[:, 0], y):
        val_percent = 100 * (val - y_min_value) / (y_max_value - y_min_value)
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f'{element}: {val} ({val_percent:.2f}%)'
        ).add_to(m)

    # Add MeasureControl for scale bar
    MeasureControl().add_to(m)

    # Add LocateControl for grid
    LocateControl().add_to(m)

    # Add Draw plugin without export button
    draw = Draw(
        draw_options={
            "polyline": True,
            "polygon": True,
            "circle": True,
            "rectangle": True,
            "marker": True,
            "circlemarker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        }
    )
    draw.add_to(m)

    # Add custom scale control
    m.get_root().html.add_child(folium.Element('<div class="folium-map-legend">Latitude</div>'))
    m.get_root().html.add_child(folium.Element('<div class="folium-map-legend" style="position: absolute; bottom: 30px;">Longitude</div>'))

    # Add color bar
    colormap.add_to(m)

    return m

# Function to create Kriging interpolated map
@st.cache_data
def create_kriging_interpolated_map(df, element):
    element_data = df[['longitude', 'latitude', element]].dropna()
    X = element_data[['longitude', 'latitude']].values
    y = element_data[element].values
    rbfi = Rbf(X[:, 0], X[:, 1], y, function='gaussian')
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  # Reduce grid resolution
    interpolated_values = rbfi(xx, yy)

    # Normalize values to percentage
    y_min_value, y_max_value = y.min(), y.max()
    interpolated_values_percent = 100 * (interpolated_values - y_min_value) / (y_max_value - y_min_value)

    # Create folium map
    m = folium.Map(location=[X[:, 1].mean(), X[:, 0].mean()], zoom_start=10)

    # Define colormap using original min and max values
    colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'red'], vmin=y_min_value, vmax=y_max_value, caption=element)

    # Create the interpolated image base64
    img_base64 = create_interpolated_image_base64(interpolated_values_percent, plt.get_cmap('plasma'))

    # Add interpolated values as a raster layer
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_base64}',
        bounds=[[y_min, x_min], [y_max, x_max]],
        opacity=0.7
    ).add_to(m)

    # Add original data points with percentage values
    for lat, lon, val in zip(X[:, 1], X[:, 0], y):
        val_percent = 100 * (val - y_min_value) / (y_max_value - y_min_value)
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f'{element}: {val} ({val_percent:.2f}%)'
        ).add_to(m)

    # Add MeasureControl for scale bar
    MeasureControl().add_to(m)

    # Add LocateControl for grid
    LocateControl().add_to(m)

    # Add Draw plugin without export button
    draw = Draw(
        draw_options={
            "polyline": True,
            "polygon": True,
            "circle": True,
            "rectangle": True,
            "marker": True,
            "circlemarker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        }
    )
    draw.add_to(m)

    # Add custom scale control
    m.get_root().html.add_child(folium.Element('<div class="folium-map-legend">Latitude</div>'))
    m.get_root().html.add_child(folium.Element('<div class="folium-map-legend" style="position: absolute; bottom: 30px;">Longitude</div>'))

    # Add color bar
    colormap.add_to(m)

    return m

# Streamlit app
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Ask to GeoChatBot", "Generate Anomaly Map"])

if option == 'Ask to GeoChatBot':
    st.title("GeoChatBot 🤖")

    @st.cache_data
    def load_documents():
        loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
        return loader.load()

    @st.cache_data
    def create_vector_store(_documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=32)
        text_chunks = text_splitter.split_documents(_documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
        return FAISS.from_documents(text_chunks, embeddings)

    documents = load_documents()
    vector_store = create_vector_store(documents)

    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 300, 'temperature': 0.08})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', retriever=vector_store.as_retriever(search_kwargs={"k": 1}), memory=memory)

    def conversation_chat(query):
        with st.spinner('Wait for a moment 😊....'):
            result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    def initialize_session_state():
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]
        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

    def display_chat_history():
        reply_container = st.container()
        container = st.container()
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Ask about geology", key='input')
                submit_button = st.form_submit_button(label='Send')
            if submit_button and user_input:
                output = conversation_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

    initialize_session_state()
    display_chat_history()

elif option == 'Generate Anomaly Map':
    df = pd.read_csv("Analytical_value_55K03.csv")
    st.title("Generate Anomaly Map 🗺️")

    container = st.container()
    with container:
        st.markdown("""
        <div style="text-align: center;">
            <p>Explore and visualize geological anomalies with our advanced interpolation techniques.</p>
            <img src="https://cdn.pixabay.com/photo/2023/09/01/18/02/eyeglasses-8227429_1280.jpg" alt="Geological Anomalies" width="60%">
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <p>Use the search bar below to enter the name of the element you want to analyze. Our system will generate interpolated anomaly maps using both IDW and Kriging methods.</p>
            <p>Zoom, pan, and draw on the maps to explore the anomalies in detail.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("###")

    with container:
        search_bar = st.text_input("Enter the name of the element for anomaly maps:", key="search_bar", placeholder="e.g., au, cu, pb")
    
    element_name = st.session_state.search_bar
    if element_name:
        st.subheader("IDW Interpolated Map")
        idw_map = create_idw_interpolated_map(df, element_name)
        folium_static(idw_map)
        
        st.subheader("Kriging Interpolated Map")
        kriging_map = create_kriging_interpolated_map(df, element_name)
        folium_static(kriging_map)

        st.subheader("Select Two Points to Measure Distance")
        st.text("Use the draw tool to place two markers on the map. Draw a line between them to measure the distance.")
