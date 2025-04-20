import streamlit as st
import pickle
import pandas as pd

@st.cache_resource 
def load_model():
    with open('best_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


model = load_model()

st.title("Prediksi Status Pemesanan Hotel")
st.sidebar.header('Input Data Pelanggan')

no_of_adults = st.sidebar.number_input('Jumlah Orang Dewasa', min_value=1, max_value=10, value=1)
no_of_children = st.sidebar.number_input('Jumlah Anak', min_value=0, max_value=10, value=0)
no_of_weekend_nights = st.sidebar.number_input('Jumlah Malam Akhir Pekan', min_value=0, max_value=7, value=1)
no_of_week_nights = st.sidebar.number_input('Jumlah Malam Hari Kerja', min_value=0, max_value=7, value=3)
required_car_parking_space = st.sidebar.selectbox('Butuh Tempat Parkir?', ['Ya', 'Tidak'])
lead_time = st.sidebar.number_input('Lead Time (Hari)', min_value=0, max_value=365, value=30)
repeated_guest = st.sidebar.selectbox('Tamu Berulang?', ['Ya', 'Tidak'])
type_of_meal_plan = st.sidebar.selectbox('Paket Makanan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
room_type_reserved = st.sidebar.selectbox('Tipe Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
arrival_year = st.sidebar.selectbox('Tahun Kedatangan', [2017]) 
arrival_month = st.sidebar.selectbox('Bulan Kedatangan', list(range(1, 13)))
arrival_date = st.sidebar.selectbox('Tanggal Kedatangan', list(range(1, 32)))
market_segment_type = st.sidebar.selectbox('Segment Pasar', ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
no_of_previous_cancellations = st.sidebar.number_input('Jumlah Pembatalan Sebelumnya', min_value=0, max_value=20, value=0)
no_of_previous_bookings_not_canceled = st.sidebar.number_input('Jumlah Booking Tidak Dibatalkan Sebelumnya', min_value=0, max_value=20, value=0)
avg_price_per_room = st.sidebar.number_input('Harga Rata-Rata per Kamar (€)', min_value=0.0, value=100.0)
no_of_special_requests = st.sidebar.number_input('Jumlah Permintaan Khusus', min_value=0, max_value=5, value=0)


required_car_parking_space = 1 if required_car_parking_space == 'Ya' else 0
repeated_guest = 1 if repeated_guest == 'Ya' else 0

input_df = pd.DataFrame({
    'no_of_adults': [no_of_adults],
    'no_of_children': [no_of_children],
    'no_of_weekend_nights': [no_of_weekend_nights],
    'no_of_week_nights': [no_of_week_nights],
    'type_of_meal_plan': [type_of_meal_plan],
    'required_car_parking_space': [required_car_parking_space],
    'room_type_reserved': [room_type_reserved],
    'lead_time': [lead_time],
    'arrival_year': [arrival_year],
    'arrival_month': [arrival_month],
    'arrival_date': [arrival_date],
    'market_segment_type': [market_segment_type],
    'repeated_guest': [repeated_guest],
    'no_of_previous_cancellations': [no_of_previous_cancellations],
    'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
    'avg_price_per_room': [avg_price_per_room],
    'no_of_special_requests': [no_of_special_requests]
})

with open('meal_plan_mapping.pkl', 'rb') as f:
    meal_plan_mapping = pickle.load(f)
input_df['type_of_meal_plan'] = input_df['type_of_meal_plan'].map(meal_plan_mapping)

with open('room_type_mapping.pkl', 'rb') as f:
    room_type_mapping = pickle.load(f)
input_df['room_type_reserved'] = input_df['room_type_reserved'].map(room_type_mapping)

with open('market_segment_mapping.pkl', 'rb') as f:
    market_segment_mapping = pickle.load(f)
input_df['market_segment_type'] = input_df['market_segment_type'].map(market_segment_mapping)

if st.sidebar.button('Prediksi'):
    prediction = model.predict(input_df)[0]
    status = 'Dibatalkan' if prediction == 1 else 'Tidak Dibatalkan'
    st.subheader('Hasil Prediksi')
    st.write(f"Status Pemesanan: **{status}**")
