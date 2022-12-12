import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from app_data import options
from app_data.predictors import batch_predict
from app_data.predictors import sample_predict

# Data Constants
COLUMNS = ['id', 'amount_tsh', 'date_recorded', 'funder', 'gps_height',
           'installer', 'longitude', 'latitude', 'wpt_name', 'num_private',
           'basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga',
           'ward', 'population', 'public_meeting', 'recorded_by',
           'scheme_management', 'scheme_name', 'permit', 'construction_year',
           'extraction_type', 'extraction_type_group', 'extraction_type_class',
           'management', 'management_group', 'payment', 'payment_type',
           'water_quality', 'quality_group', 'quantity', 'quantity_group',
           'source', 'source_type', 'source_class', 'waterpoint_type',
           'waterpoint_type_group']

# Medians and modes
GPS_HEIGHT_MEDIAN = 369.0
LATITUDE_MEDIAN = -5.0216
LONGITUDE_MEDIAN = 34.9087
CONSTRUCTION_YEAR_MEDIAN = 2010
PUBLIC_MEETING_MODE = True
SCHEME_MANAGEMENT_MODE = "VWC"
PERMIT_MODE = True
POPULATION_MEDIAN = 0
BASIN_MODE = "Lake Victoria"
LGA_MODE = "Njombe"
REGION_MODE = "Iringa"
EXTRACTION_TYPE_CLASS_MODE = "gravity"
MANAGEMENT_MODE = "vwc"
PAYMENT_TYPE_MODE = "never pay"
WATER_QUALITY_MODE = "soft"
QUANTITY_MODE = "enough"
SOURCE_TYPE_MODE = "spring"
SOURCE_CLASS_MODE = "groundwater"
WATERPOINT_TYPE_MODE = "communal standpipe"

# App constants
BASIN_OPTIONS = options.BASIN_OPTIONS
REGION_OPTIONS = options.REGION_OPTIONS
LGA_OPTIONS = options.LGA_OPTIONS
PUBLIC_MEETING_OPTIONS = options.PUBLIC_MEETING_OPTIONS
PERMIT_OPTIONS = options.PERMIT_OPTIONS
SCHEME_MANAGEMENT_OPTIONS = options.SCHEME_MANAGEMENT_OPTIONS
EXTRACTION_TYPE_CLASS_OPTIONS = options.EXTRACTION_TYPE_CLASS_OPTIONS
MANAGEMENT_OPTIONS = options.MANAGEMENT_OPTIONS
PAYMENT_TYPE_OPTIONS = options.PAYMENT_TYPE_OPTIONS
WATER_QUALITY_OPTIONS = options.WATER_QUALITY_OPTIONS
QUANTITY_OPTIONS = options.QUANTITY_OPTIONS
SOURCE_TYPE_OPTIONS = options.SOURCE_TYPE_OPTIONS
SOURCE_CLASS_OPTIONS = options.SOURCE_CLASS_OPTIONS
WATERPOINT_TYPE_OPTIONS = options.WATERPOINT_TYPE_OPTIONS


# Convert dataframe to csv
# @st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


# Application main function
def main():
    """Application main function
    """
    st.title("Water Pumps Functionality Status Prediction App")
    st.write("A Web Application to predict the functionality status of water pumps installed across Tanzania.")

    with st.sidebar:

        selection = option_menu("Prediction Options",
                                ['Prediction for Single Pump',
                                 'Prediction for Multiple Pumps'],
                                icons=['check2', 'check2-all'],
                                menu_icon="laptop", default_index=0)

    if selection == 'Prediction for Single Pump':
        st.header("Single Pump Prediction")
        st.subheader("Input relevant data in the following fields:")
        predict_button_status = True

        with st.form("Input values"):

            # Columns in app
            col1, col2 = st.columns(2)

            with col1:
                input_gps_height = float(st.number_input(
                    "GPS Height", min_value=0.0, value=GPS_HEIGHT_MEDIAN))

                input_population = st.number_input(
                    "Population", min_value=0, value=POPULATION_MEDIAN)

                input_construction_year = int(st.number_input(
                    "Construction year", min_value=1960, max_value=2013, value=CONSTRUCTION_YEAR_MEDIAN, step=1))

                input_basin = st.selectbox("Basin", options=BASIN_OPTIONS)
                if input_basin == None:
                    input_basin = BASIN_MODE

                input_lga = st.selectbox("lga", options=LGA_OPTIONS)
                if input_lga == None:
                    input_lga = LGA_MODE

                input_scheme_management = st.selectbox(
                    "Scheme Managemnt", options=SCHEME_MANAGEMENT_OPTIONS)
                if input_scheme_management == None:
                    input_scheme_management = SCHEME_MANAGEMENT_MODE

                input_managemnt = st.selectbox(
                    "Management", options=MANAGEMENT_OPTIONS)
                if input_managemnt == None:
                    input_managemnt = MANAGEMENT_MODE

                input_water_quality = st.selectbox(
                    "Water Quality", options=WATER_QUALITY_OPTIONS)
                if input_water_quality == None:
                    input_water_quality = WATER_QUALITY_MODE

                input_source_type = st.selectbox(
                    "Source Type", options=SOURCE_TYPE_OPTIONS)
                if input_source_type == None:
                    input_source_type = SOURCE_TYPE_MODE

                input_waterpoint_type = st.selectbox(
                    "Waterpoint Type", options=WATERPOINT_TYPE_OPTIONS)
                if input_waterpoint_type == None:
                    input_waterpoint_type = WATERPOINT_TYPE_MODE

            with col2:
                input_longitude = st.number_input(
                    "Longitude", min_value=28.0, max_value=41.0, value=LONGITUDE_MEDIAN)

                input_latitude = st.number_input(
                    "Latitude", min_value=-13.0, max_value=0.0, value=LATITUDE_MEDIAN)

                input_region = st.selectbox("Region", options=REGION_OPTIONS)
                if input_region == None:
                    input_region = REGION_MODE

                input_public_meeting = st.selectbox("Public Meeting",
                                                    options=PUBLIC_MEETING_OPTIONS)
                if input_public_meeting == None:
                    input_public_meeting = PUBLIC_MEETING_MODE

                input_permit = st.selectbox("Permit",
                                            options=PERMIT_OPTIONS)
                if input_permit == None:
                    input_permit = PERMIT_MODE

                input_extraction_type_class = st.selectbox(
                    "Extraction Type Class", options=EXTRACTION_TYPE_CLASS_OPTIONS)
                if input_extraction_type_class == None:
                    input_extraction_type_class = EXTRACTION_TYPE_CLASS_MODE

                input_payment_type = st.selectbox(
                    "Payment Type", options=PAYMENT_TYPE_OPTIONS)
                if input_payment_type == None:
                    input_payment_type = PAYMENT_TYPE_MODE

                input_quantity = st.selectbox(
                    "Quantity", options=QUANTITY_OPTIONS)
                if input_quantity == None:
                    input_quantity = QUANTITY_MODE

                input_source_class = st.selectbox(
                    "Source Class", options=SOURCE_CLASS_OPTIONS)
                if input_source_class == None:
                    input_source_class = SOURCE_CLASS_MODE

                input_date_recorded = st.text_input(
                    "Date Recorded (DD-MM-YYYY)", value="DD-MM-YYYY")

            # Every form must have a submit button.
            if st.form_submit_button("Submit"):
                if input_date_recorded == "DD-MM-YYYY":
                    st.error("Please enter date in required format")
                else:
                    st.success("Received data succesfully.",  icon="✅")
                    st.success(
                        "Press button below to get pump status.",  icon="✅")
                    predict_button_status = False

        # Create input sample
        input_sample = np.array([0, 0, input_date_recorded, "funder", input_gps_height,
                                 "installer", input_longitude, input_latitude,
                                 "wpt_name", 0, input_basin, "subvillage", input_region,
                                 "rcode", "dcode", input_lga, "ward", input_population,
                                 input_public_meeting, "recorder", input_scheme_management,
                                 "scheme_name", input_permit, input_construction_year,
                                 "ext_type", "ext_grp", input_extraction_type_class,
                                 input_managemnt, "mng_grp", "payment", input_payment_type,
                                 input_water_quality, "quality_grp", input_quantity,
                                 "qty_grp", "src", input_source_type, input_source_class,
                                 input_waterpoint_type, "wpt_tp_grp"]).reshape(1, -1)

        if st.button('Predict Pump Status', disabled=predict_button_status):
            prediction = sample_predict(input_sample, return_id=False)
            pred = str(prediction['status_group'].values[0])
            st.info(f'Pump status: {pred}', icon="✔")

    elif selection == 'Prediction for Multiple Pumps':
        st.header("Batch Prediction")
        st.write(
            "Upload data.csv file having batch of records and get predictions as predictions.csv")

        st.write("Your data file should be like sample data file.")
        st.write("Sample Data File:")
        data = pd.read_csv("data\data_sample.csv")
        data_sample = pd.DataFrame(data, columns=COLUMNS)
        st.dataframe(data_sample)
        input_file = st.file_uploader(label='Please upload data file in .csv format',
                                      type='.csv', accept_multiple_files=False)

        # code for Prediction
        # creating a button for Prediction
        if st.button('Get Predictions'):
            input_data = pd.read_csv(input_file, parse_dates=[
                                     'date_recorded'], dayfirst=True)
            predictions = batch_predict(input_data, return_id=True)
            csv = convert_df(predictions)
            st.success(
                'Success..! You can Download predictions.csv using button below.', icon="✅")
            st.download_button("Download Predictions file", csv,
                               "predictions.csv", "text/csv", key='download-csv')


if __name__ == '__main__':
    main()
