import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import streamlit as st



# path constant
TRANSFORM_FILE_PATH = "ml\data_transform"
MODEL_FILE_PATH = "ml\model\ml_model.joblib"


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


# Load model
@st.cache(allow_output_mutation=True)
def load_model(model_filepath):
    """Loads ml model in memory

    Args:
        model_filepath (str): File path of the ml model

    Returns:
        scikit-learn ml model object
    """
    model = joblib.load(MODEL_FILE_PATH)
    return model


# Prediction pipeline function for batch
def batch_predict(input_samples: any, return_id=False):
    """Receives raw input data and returns predictions given by ml model

    Args:
        input_samples (any): Raw input sample/s
        return_id (bool, optional): True to return input sample row ids. Defaults to False.

    Returns:
        pd.DataFrame: Predictions dataframe
    """
    # Columns names in data
    columns = COLUMNS

    # Create dataframe from input samples
    input_df = pd.DataFrame(input_samples, columns=columns)

    # Imputing missing values
    # gps_height
    input_df['gps_height'] = np.where(
        input_df['gps_height'] <= 0, GPS_HEIGHT_MEDIAN, input_df['gps_height'])
    # latitude
    input_df['latitude'] = np.where(
        input_df['latitude'] < -11.8, LATITUDE_MEDIAN, input_df['latitude'])
    input_df['latitude'] = np.where(
        input_df['latitude'] > -1, LATITUDE_MEDIAN, input_df['latitude'])
    # longitude
    input_df['longitude'] = np.where(
        input_df['longitude'] < 29, LONGITUDE_MEDIAN, input_df['longitude'])
    input_df['longitude'] = np.where(
        input_df['longitude'] > 40.8, LONGITUDE_MEDIAN, input_df['longitude'])
    # public_meeting
    input_df['public_meeting'] = input_df['public_meeting'].fillna(
        PUBLIC_MEETING_MODE)
    # scheme_management
    input_df['scheme_management'] = input_df['scheme_management'].fillna(
        SCHEME_MANAGEMENT_MODE)
    # permit
    input_df['permit'] = input_df['permit'].fillna(PERMIT_MODE)
    # construction_year

    def year_imputer(x):
        """
        Function to impute construction_year in data
        """
        if x == 0:
            year_range = list(range(2000, 2013))
            impute_year = np.random.choice(year_range)
            return impute_year

    input_df['construction_year'] = np.where(input_df['construction_year'] == 0,
                                             input_df['construction_year'].apply(
                                                 year_imputer),
                                             input_df['construction_year'])

    # Feature Engineering
    # Creating separate year and month columns
    input_df['date_recorded'] = pd.to_datetime(
        input_df['date_recorded'], dayfirst=True)
    input_df['record_year'] = input_df['date_recorded'].dt.year
    input_df['record_month'] = input_df['date_recorded'].dt.month
    input_df['waterpoint_age'] = input_df['record_year'] - \
        input_df['construction_year']

    # treating record_year and record_month as categorical features
    input_df[['record_year', 'record_month']] = input_df[[
        'record_year', 'record_month']].astype('object')

    # Feature Selection
    ids = input_df['id'].values
    dropped_features = ['id', 'amount_tsh', 'funder', 'installer', 'wpt_name',
                        'num_private', 'subvillage', 'region_code', 'district_code',
                        'ward', 'recorded_by', 'scheme_name', 'construction_year',
                        'extraction_type', 'extraction_type_group', 'management_group',
                        'payment', 'quality_group', 'quantity_group', 'source',
                        'waterpoint_type_group', 'date_recorded']
    input_df = input_df.drop(dropped_features, axis=1)

    # Data Preparation

    # gps_height
    gps_height_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "gps_height_normalizer"))
    test_gps_height_normalized = gps_height_normalizer.transform(
        input_df['gps_height'].values.reshape(-1, 1))
    # longitude
    longitude_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "longitude_normalizer"))
    test_longitude_normalized = longitude_normalizer.transform(
        input_df['longitude'].values.reshape(-1, 1))
    # latitude
    latitude_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "latitude_normalizer"))
    test_latitude_normalized = latitude_normalizer.transform(
        input_df['latitude'].values.reshape(-1, 1))
    # population
    population_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "population_normalizer"))
    test_population_normalized = population_normalizer.transform(
        input_df['population'].values.reshape(-1, 1))
    # waterpoint_age
    waterpoint_age_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "waterpoint_age_normalizer"))
    test_waterpoint_age_normalized = waterpoint_age_normalizer.transform(
        input_df['waterpoint_age'].values.reshape(-1, 1))
    # basin
    basin_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "basin_encoder"))
    test_basin_encoded = basin_encoder.transform(
        input_df['basin'].values.reshape(-1, 1))
    # Region
    region_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "region_encoder"))
    test_region_encoded = region_encoder.transform(
        input_df['region'].values.reshape(-1, 1))
    # lga
    lga_encoder = joblib.load(os.path.join(TRANSFORM_FILE_PATH, "lga_encoder"))
    test_lga_encoded = lga_encoder.transform(
        input_df['lga'].values.reshape(-1, 1))
    # public_meeting
    public_meeting_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "public_meeting_encoder"))
    test_public_meeting_encoded = public_meeting_encoder.transform(
        input_df['public_meeting'].values.reshape(-1, 1))
    # scheme_management
    scheme_management_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "scheme_management_encoder"))
    test_scheme_management_encoded = scheme_management_encoder.transform(
        input_df['scheme_management'].values.reshape(-1, 1))
    # permit
    permit_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "permit_encoder"))
    test_permit_encoded = permit_encoder.transform(
        input_df['permit'].values.reshape(-1, 1))
    # extraction_type_class
    extraction_type_class_encoder = joblib.load(os.path.join(TRANSFORM_FILE_PATH,
                                                             "extraction_type_class_encoder"))
    test_extraction_type_class_encoded = extraction_type_class_encoder.transform(
        input_df['extraction_type_class'].values.reshape(-1, 1))
    # management
    management_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "management_encoder"))
    test_management_encoded = management_encoder.transform(
        input_df['management'].values.reshape(-1, 1))
    # payment_type
    payment_type_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "payment_type_encoder"))
    test_payment_type_encoded = payment_type_encoder.transform(
        input_df['payment_type'].values.reshape(-1, 1))
    # water_quality
    water_quality_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "water_quality_encoder"))
    test_water_quality_encoded = water_quality_encoder.transform(
        input_df['water_quality'].values.reshape(-1, 1))
    # quantity
    quantity_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "quantity_encoder"))
    test_quantity_encoded = quantity_encoder.transform(
        input_df['quantity'].values.reshape(-1, 1))
    # source_type
    source_type_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "source_type_encoder"))
    test_source_type_encoded = source_type_encoder.transform(
        input_df['source_type'].values.reshape(-1, 1))
    # source_class
    source_class_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "source_class_encoder"))
    test_source_class_encoded = source_class_encoder.transform(
        input_df['source_class'].values.reshape(-1, 1))
    # waterpoint_type
    waterpoint_type_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "waterpoint_type_encoder"))
    test_waterpoint_type_encoded = waterpoint_type_encoder.transform(
        input_df['waterpoint_type'].values.reshape(-1, 1))
    # record_year
    record_year_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "record_year_encoder"))
    test_record_year_encoded = record_year_encoder.transform(
        input_df['record_year'].values.reshape(-1, 1))
    # record_month
    record_month_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "record_month_encoder"))
    test_record_month_encoded = record_month_encoder.transform(
        input_df['record_month'].values.reshape(-1, 1))

    # Creating data matrix
    input_encoded = hstack([test_gps_height_normalized, test_longitude_normalized, test_latitude_normalized,
                            test_basin_encoded, test_region_encoded, test_lga_encoded, test_population_normalized,
                            test_public_meeting_encoded, test_scheme_management_encoded, test_permit_encoded,
                            test_extraction_type_class_encoded, test_management_encoded, test_payment_type_encoded,
                            test_water_quality_encoded, test_quantity_encoded, test_source_type_encoded,
                            test_source_class_encoded, test_waterpoint_type_encoded, test_record_year_encoded,
                            test_record_month_encoded, test_waterpoint_age_normalized]).tocsr()

    # Load model
    model = load_model(MODEL_FILE_PATH)

    # predictions
    predictions = model.predict(input_encoded)
    label_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "label_encoder"))
    predictions = label_encoder.inverse_transform(predictions)

    if return_id:
        pred_dict = {"id": ids, "status_group": predictions}
    else:
        pred_dict = {"status_group": predictions}

    pred_df = pd.DataFrame(pred_dict)
    return pred_df


# Prediction function for single sample
def sample_predict(input_samples: any, return_id=False):
    """Receives raw input data and returns predictions given by ml model

    Args:
        input_samples (any): Raw input sample/s
        return_id (bool, optional): True to return input sample row ids. Defaults to False.

    Returns:
        pd.DataFrame: Predictions dataframe
    """
    # Columns names in data
    columns = COLUMNS

    # Create dataframe from input samples
    input_df = pd.DataFrame(input_samples, columns=columns)

    # Feature Engineering
    # Creating separate year and month columns
    input_df['date_recorded'] = pd.to_datetime(
        input_df['date_recorded'], dayfirst=True)
    input_df['record_year'] = input_df['date_recorded'].dt.year
    input_df['record_month'] = input_df['date_recorded'].dt.month
    input_df['waterpoint_age'] = input_df['record_year'] - \
        int(input_df['construction_year'])

    # treating record_year and record_month as categorical features
    input_df[['record_year', 'record_month']] = input_df[[
        'record_year', 'record_month']].astype('object')

    # Feature Selection
    ids = input_df['id'].values
    dropped_features = ['id', 'amount_tsh', 'funder', 'installer', 'wpt_name',
                        'num_private', 'subvillage', 'region_code', 'district_code',
                        'ward', 'recorded_by', 'scheme_name', 'construction_year',
                        'extraction_type', 'extraction_type_group', 'management_group',
                        'payment', 'quality_group', 'quantity_group', 'source',
                        'waterpoint_type_group', 'date_recorded']
    input_df = input_df.drop(dropped_features, axis=1)

    # Data Preparation

    # gps_height
    gps_height_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "gps_height_normalizer"))
    test_gps_height_normalized = gps_height_normalizer.transform(
        input_df['gps_height'].values.reshape(-1, 1))
    # longitude
    longitude_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "longitude_normalizer"))
    test_longitude_normalized = longitude_normalizer.transform(
        input_df['longitude'].values.reshape(-1, 1))
    # latitude
    latitude_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "latitude_normalizer"))
    test_latitude_normalized = latitude_normalizer.transform(
        input_df['latitude'].values.reshape(-1, 1))
    # population
    population_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "population_normalizer"))
    test_population_normalized = population_normalizer.transform(
        input_df['population'].values.reshape(-1, 1))
    # waterpoint_age
    waterpoint_age_normalizer = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "waterpoint_age_normalizer"))
    test_waterpoint_age_normalized = waterpoint_age_normalizer.transform(
        input_df['waterpoint_age'].values.reshape(-1, 1))
    # basin
    basin_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "basin_encoder"))
    test_basin_encoded = basin_encoder.transform(
        input_df['basin'].values.reshape(-1, 1))
    # Region
    region_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "region_encoder"))
    test_region_encoded = region_encoder.transform(
        input_df['region'].values.reshape(-1, 1))
    # lga
    lga_encoder = joblib.load(os.path.join(TRANSFORM_FILE_PATH, "lga_encoder"))
    test_lga_encoded = lga_encoder.transform(
        input_df['lga'].values.reshape(-1, 1))
    # public_meeting
    public_meeting_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "public_meeting_encoder"))
    test_public_meeting_encoded = public_meeting_encoder.transform(
        input_df['public_meeting'].values.reshape(-1, 1))
    # scheme_management
    scheme_management_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "scheme_management_encoder"))
    test_scheme_management_encoded = scheme_management_encoder.transform(
        input_df['scheme_management'].values.reshape(-1, 1))
    # permit
    permit_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "permit_encoder"))
    test_permit_encoded = permit_encoder.transform(
        input_df['permit'].values.reshape(-1, 1))
    # extraction_type_class
    extraction_type_class_encoder = joblib.load(os.path.join(TRANSFORM_FILE_PATH,
                                                             "extraction_type_class_encoder"))
    test_extraction_type_class_encoded = extraction_type_class_encoder.transform(
        input_df['extraction_type_class'].values.reshape(-1, 1))
    # management
    management_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "management_encoder"))
    test_management_encoded = management_encoder.transform(
        input_df['management'].values.reshape(-1, 1))
    # payment_type
    payment_type_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "payment_type_encoder"))
    test_payment_type_encoded = payment_type_encoder.transform(
        input_df['payment_type'].values.reshape(-1, 1))
    # water_quality
    water_quality_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "water_quality_encoder"))
    test_water_quality_encoded = water_quality_encoder.transform(
        input_df['water_quality'].values.reshape(-1, 1))
    # quantity
    quantity_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "quantity_encoder"))
    test_quantity_encoded = quantity_encoder.transform(
        input_df['quantity'].values.reshape(-1, 1))
    # source_type
    source_type_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "source_type_encoder"))
    test_source_type_encoded = source_type_encoder.transform(
        input_df['source_type'].values.reshape(-1, 1))
    # source_class
    source_class_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "source_class_encoder"))
    test_source_class_encoded = source_class_encoder.transform(
        input_df['source_class'].values.reshape(-1, 1))
    # waterpoint_type
    waterpoint_type_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "waterpoint_type_encoder"))
    test_waterpoint_type_encoded = waterpoint_type_encoder.transform(
        input_df['waterpoint_type'].values.reshape(-1, 1))
    # record_year
    record_year_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "record_year_encoder"))
    test_record_year_encoded = record_year_encoder.transform(
        input_df['record_year'].values.reshape(-1, 1))
    # record_month
    record_month_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "record_month_encoder"))
    test_record_month_encoded = record_month_encoder.transform(
        input_df['record_month'].values.reshape(-1, 1))

    # Creating data matrix
    input_encoded = hstack([test_gps_height_normalized, test_longitude_normalized, test_latitude_normalized,
                            test_basin_encoded, test_region_encoded, test_lga_encoded, test_population_normalized,
                            test_public_meeting_encoded, test_scheme_management_encoded, test_permit_encoded,
                            test_extraction_type_class_encoded, test_management_encoded, test_payment_type_encoded,
                            test_water_quality_encoded, test_quantity_encoded, test_source_type_encoded,
                            test_source_class_encoded, test_waterpoint_type_encoded, test_record_year_encoded,
                            test_record_month_encoded, test_waterpoint_age_normalized]).tocsr()

    # Load model
    model = load_model(MODEL_FILE_PATH)

    # predictions
    predictions = model.predict(input_encoded)
    label_encoder = joblib.load(os.path.join(
        TRANSFORM_FILE_PATH, "label_encoder"))
    predictions = label_encoder.inverse_transform(predictions)

    if return_id:
        pred_dict = {"id": ids, "status_group": predictions}
    else:
        pred_dict = {"status_group": predictions}

    pred_df = pd.DataFrame(pred_dict)
    return pred_df