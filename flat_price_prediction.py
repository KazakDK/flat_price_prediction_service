import streamlit as st
from PIL import Image
import pandas as pd
from pickle import load
import sklearn


first_img = Image.open('domik.jpeg')

st.set_page_config(
    layout='wide',
    page_title='',
    page_icon=first_img)

st.title('Квартирный вопрос')
st.image(first_img,width=500)

st.sidebar.header('Параметры квартиры:')
df = pd.read_csv('clean_flats_data.csv')

rooms = st.sidebar.selectbox('Кол-во комнат',(sorted(df['rooms'].unique())))

area = st.sidebar.slider('Площадь квартиры', min_value=round(df['area'].min()),
                             max_value=round(df['area'].max()),
                             value=round(df['area'].min()), step=1)

living_area = st.sidebar.slider('Жилая площадь', min_value=round(df['living_area'].min()),
                             max_value=round(df['living_area'].max()),
                             value=round(df['living_area'].min()), step=1)

kitchen_area = st.sidebar.slider('Площадь кухни', min_value=round(df['kitchen_area'].min()),
                             max_value=round(df['kitchen_area'].max()),
                             value=round(df['kitchen_area'].min()), step=1)

floor = st.sidebar.selectbox('Этаж', (sorted(df['floor'].unique())))

finishing = st.sidebar.selectbox('Ремонт', (sorted(df['finishing'].unique())))

deal_type = st.sidebar.selectbox('Тип сделки', (sorted(df['deal_type'].unique())))

house_age = st.sidebar.slider('Год постройки', min_value=round(df['house_age'].min()),
                             max_value=round(df['house_age'].max()),
                             value=round(df['house_age'].min()), step=1)

wall_material =  st.sidebar.selectbox('Материал стен', (sorted(df['wall_material'].unique())))

default_house_value = sorted(df['house_series'].unique())
default_ix = default_house_value.index('Серия неизвестна')
house_series = st.sidebar.selectbox('Серия дома', default_house_value, index=default_ix)

house_floors = st.sidebar.slider('Этажность дома', min_value=round(df['house_floors'].min()),
                             max_value=round(df['house_floors'].max()),
                             value=round(df['house_floors'].min()), step=1)

elevator = st.sidebar.selectbox('Наличие лифта', sorted(df['elevator'].unique()))

latitude = st.sidebar.text_input('Широта')
longitude = st.sidebar.text_input('Долгота')

center_distance = st.sidebar.slider('Расстояние до центра', min_value=round(df['center_distance'].min()),
                             max_value=round(df['center_distance'].max()),
                             value=round(df['center_distance'].min()), step=1)

guest_data = {
'rooms' : float(rooms),
'area' : float(area),
'living_area' : float(living_area),
'kitchen_area' : float(kitchen_area),
'floor' : float(floor),
'finishing' : finishing,
'deal_type' : deal_type,
'house_age' : float(house_age),
'wall_material' : wall_material,
'house_series': house_series,
'house_floors': float(house_floors),
'elevator': elevator,
'latitude': float(latitude.replace('','55.719597')),
'longitude': float(longitude.replace('','55.719597')),
'center_distance': float(center_distance)
}

res_df = pd.DataFrame(guest_data,index=[0])

st.write('''
## Хотите выгодно продать или купить квартиру?
Вы можете выбрать параметры квартиры в меню выбора параметров, оно находится слева,\n
и мы скажем вам со 100% точностью его стоимость!''')

st.write(res_df)


with open('model.pickle','rb') as file:
    model = load(file)
with open('encoder.pickle','rb') as file:
    encoder = pd.read_pickle(file)
with open('scal.pickle','rb') as file:
    scal = load(file)

encoded_res = encoder.transform(res_df)
scaled_res = scal.transform(encoded_res)
overall_pred = model.predict(scaled_res)
st.write(f'Стоимость квартиры вашей мечты: {round(overall_pred[0])}')