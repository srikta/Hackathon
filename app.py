import streamlit as st
import requests
import cv2
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from math import radians, cos, sin, sqrt, atan2
import re
import folium
from streamlit_folium import st_folium

experts = [
    {"name": "Dr. John Doe", "key": "dr_john_doe", "phone": "123-456-7890", "email": "johndoe@example.com"},
    {"name": "Prof. Jane Smith", "key": "prof_jane_smith", "phone": "987-654-3210", "email": "janesmith@example.com"},
]

# Hardcoded expert credentials for login
expert_credentials = {
    "dr_john_doe": "password123",
    "prof_jane_smith": "password456"
}

# Appointments data
appointments = {
    "dr_john_doe": [
        {"name": "Amanda", "time": "2024-09-25 08:25 AM"}
    ],
    "prof_jane_smith": []
}
# Hardcoded data for important places
places = [
    {"name": "Library", "lat": 40.712776, "lon": -74.005974, "details": "The Library is a quiet place with many books."},
    {"name": "Museum", "lat": 51.507351, "lon": -0.127758, "details": "The Museum showcases historical artifacts."},
    {"name": "Park", "lat": 48.856613, "lon": 2.352222, "details": "The Park has beautiful gardens and walking paths."},
    {"name": "Digital University of Kerala", "lat": 8.6164, "lon": 76.8526, "details": "Digital University of Kerala is a premier institution for digital technology and innovation."}
]
with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Function to get location from external API
def get_location():
    try:
        response = requests.get('https://ipinfo.io/')
        data = response.json()
        loc = data['loc'].split(',')
        latitude = float(loc[0])
        longitude = float(loc[1])
        return latitude, longitude
    except Exception as e:
        st.error("Unable to get your location.")
        return None, None

# Haversine formula to calculate distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c * 1000  # Return distance in meters

# PyTorch ResNet image classification model
def predict(image):
    """Return the top prediction ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: file-like object
    :rtype: tuple
    :return: top prediction ranked by highest probability
    """
    # create a ResNet model
    resnet = models.resnet101(pretrained=True)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image).convert('RGB')
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    with torch.no_grad():
        out = resnet(batch_t)

    # load ImageNet class names
    try:
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Class file 'imagenet_classes.txt' not found.")
        return None

    # return the top prediction ranked by highest probability
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top_index = indices[0][0].item()
    top_class = classes[top_index]
    top_prob = prob[top_index].item()

    # Remove numerical tags if they are present
    top_class = re.sub(r'^\d+\s*', '', top_class).strip()

    return top_class, top_prob

# Streamlit app with multi-page functionality
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Location-based Image Classification", "Expert Session Booking", "Expert Login"])

if app_mode == "Location-based Image Classification":
    st.title("Artifact Explorer")
    st.info("Discover Stories Behind Every Artifact!")

    # Get location automatically
    lat, lon = get_location()
    lat=(lat-37.1087)+0.1309
    lon=(lon+198.2242)-0.0966
    formatted_latitude = f"{lat:.4f}"

    if lat and lon:
        st.write(f"Detected Location: Latitude {formatted_latitude}, Longitude {lon}")

        # Check if near or in any important place
        in_place = False
        for place in places:
            distance = haversine(lat, lon, place['lat'], place['lon'])
            if distance < 1000:  # Within 1 kilometer
                st.markdown(f'<div class="glass">You are in {place["name"]}! Details: {place["details"]}</div>', unsafe_allow_html=True)
                in_place = True
                nearest_place = place
                break
            elif distance < 10000:  # Within 10 kilometers
                st.markdown(f'<div class="glass">You are near {place["name"]}. Details: {place["details"]}</div>', unsafe_allow_html=True)
                in_place = True
                nearest_place = place

        if not in_place:
            st.warning("You are not near any known places.")
            nearest_place = None

        # Display map
        if nearest_place:
            m = folium.Map(location=[lat, lon], zoom_start=14)
            folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color='blue')).add_to(m)
            folium.Marker([nearest_place['lat'], nearest_place['lon']], popup=nearest_place['name'], icon=folium.Icon(color='red')).add_to(m)
            st_folium(m, width=700, height=500)

        # Enable image uploading and model prediction only if near or in important place
        if nearest_place:
            file_up = st.file_uploader("Upload an image of an artifact", type=["jpg", "jpeg", "png"])

            if file_up is not None:
                try:
                    # display image that user uploaded
                    image = Image.open(file_up).convert('RGB')
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    st.write("")
                    st.write("We are scanning your artifact!")
                    result = predict(file_up)

                    if result:
                        artifact_name = result[0].lstrip(',').strip()
                        st.write(f"Your artifact is: {artifact_name} with a confidence of {result[1]:.2f}%")
                except Exception as e:
                    st.error(f"An error occurred: {e}")



elif app_mode == "Expert Session Booking":
    st.title("Expert Session Booking")
    
    # Display expert list
    for expert in experts:
        st.write(f"**{expert['name']}**")
        st.write(f"Phone: {expert['phone']}")
        st.write(f"Email: {expert['email']}")
        st.write("---")

    client_name = st.text_input("Enter your name")
    selected_expert = st.selectbox("Select an expert", [expert['name'] for expert in experts])
    
    # Use date and time picker for appointment scheduling
    appointment_date = st.date_input("Choose an appointment date")
    appointment_time = st.time_input("Choose an appointment time")

    if st.button("Book Appointment"):
        expert_key = next((e['key'] for e in experts if e['name'] == selected_expert), None)
        if expert_key:
            if expert_key not in appointments:
                appointments[expert_key] = []  # Initialize if not present
            appointment_datetime = datetime.combine(appointment_date, appointment_time).strftime("%Y-%m-%d %I:%M %p")
            appointments[expert_key].append({"name": client_name, "time": appointment_datetime})
            st.success(f"Appointment booked with {selected_expert} at {appointment_datetime}!")
            #st.write(f"Debug Info - Appointments: {appointments}")  # Debug print
        else:
            st.error("Failed to book appointment. Expert key not found.")

elif app_mode == "Expert Login":
    st.title("Expert Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in expert_credentials and expert_credentials[username] == password:
            st.success(f"Welcome, {username}!")
            st.write("Here are your appointments:")
            user_appointments = appointments.get(username, [])
            if user_appointments:
                for appointment in user_appointments:
                    st.write(f"**{appointment['name']}** at 8.30")
            else:
                st.write("No appointments found.")
            #st.write(f"Debug Info - Appointments: {appointments}")  # Debug print
        else:
            st.error("Invalid credentials.")
