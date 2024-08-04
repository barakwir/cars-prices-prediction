import pandas as pd
import numpy as np
import pickle

# טעינת המילונים
with open('year_km_mean_dict.pkl', 'rb') as f:
    year_km_mean_dict = pickle.load(f)

with open('overall_km_mean.pkl', 'rb') as f:
    overall_km_mean = pickle.load(f)

# מילון עם סטטיסטיקות של מודלים והמדינות שלהם, עדכן בהתאם לנתונים שלך
model_median_dict = {"model_example": 1000}
overall_median = 1200

def map_location_to_region(location):
    north = ['חיפה וחוף הכרמל', 'כרמיאל והסביבה', 'גליל ועמקים', 'עכו - נהריה', 
             'טבריה והסביבה', 'קריות', 'גליל', 'טבריה', 'זכרון - בנימינה', 
             'יישובי השומרון', 'פרדס חנה - כרכור', 'חדרה וישובי עמק חפר', 
             'מושבים בצפון']
    
    center = ['רעננה - כפר סבא', 'מושבים בשרון', 'רמת', 'נס ציונה - רחובות', 
              'ראשל"צ והסביבה', 'פתח תקוה והסביבה', 'חולון - בת ים', 
              'ירושלים והסביבה', 'מושבים', 'גדרה יבנה והסביבה', 'רמת גן - גבעתיים', 
              'תל אביב', 'ראש העין והסביבה', 'נתניה והסביבה', 'בקעת אונו', 
              'מודיעין והסביבה', 'פרדס', 'הוד השרון והסביבה', 'רמת השרון - הרצליה', 
              'חולון', 'אזור השרון והסביבה', 'מושבים במרכז', 'קיסריה והסביבה', 
              'רעננה', 'רמלה - לוד', 'תל', 'הוד', 'עמק', 'ירושלים', 'פתח', 
              'מודיעין', 'רמלה', 'ראשל"צ', 'נתניה', 'רחובות']
    
    south = ['באר שבע והסביבה', 'אשדוד - אשקלון', 'אילת והערבה', 'מושבים בדרום']
    
    unknown = ['nan', 'None']
    
    if location in north:
        return 'צפון'
    elif location in center:
        return 'מרכז'
    elif location in south:
        return 'דרום'
    elif location in unknown:
        return 'לא ידוע'
    else:
        return 'מזרח'  # General assumption for another category

def clean_model(model):
    # If the model starts with 'סקודה', remove 'סקודה ' and any years afterwards
    if model.startswith('סקודה '):
        model = model.replace('סקודה ', '', 1)  # Remove 'סקודה ' only
        model = model.split()[0]  # Take the first part of the string (model name only)
    elif model.startswith('אאודי '):
        model = model.replace('אאודי ', '', 1)  # Remove 'אאודי ' only
    elif model.startswith('מאזדה '):
        model = model.split()[1]
    elif model.startswith('רנו '):
        model = model.split()[1]
    model = model.strip()
    return model

def prepare_data(test_data, train_columns):
    # Remove duplicates
    test_data = test_data.drop_duplicates()

    # Change column types
    test_data['manufactor'] = test_data['manufactor'].astype(str)
    test_data['Year'] = test_data['Year'].astype(int)
    test_data['model'] = test_data['model'].astype(str)
    test_data['Hand'] = test_data['Hand'].astype(int)
    test_data['Gear'] = pd.Categorical(test_data['Gear'], categories=['אוטומט', 'ידני', 'רובוטית', 'טפיטרוניק'])
    test_data['capacity_Engine'] = test_data['capacity_Engine'].astype(float)
    test_data['Engine_type'] = pd.Categorical(test_data['Engine_type'], categories=['בנזין', 'היבריד', 'דיזל', 'חשמלי', 'גז'])
    test_data['Prev_ownership'] = pd.Categorical(test_data['Prev_ownership'])
    test_data['Curr_ownership'] = pd.Categorical(test_data['Curr_ownership'])
    test_data['Area'] = test_data['Area'].astype(str) if 'Area' in test_data.columns else ''
    test_data['City'] = test_data['City'].astype(str) if 'City' in test_data.columns else ''
    test_data['Description'] = test_data['Description'].astype(str)
    test_data['Color'] = test_data['Color'].astype(str)
    test_data['Km'] = pd.to_numeric(test_data['Km'], errors='coerce').astype('float64')

    # הדפסת עמודות לאימות
    print("Columns in test_data before processing:")
    print(test_data.columns)

    # Fill nulls in Gear
    test_data['Gear'].fillna('אוטומט', inplace=True)

    # Merge same values
    test_data['Gear'] = test_data['Gear'].replace('אוטומטית', 'אוטומט')

    # Fill nulls in capacity_Engine
    missing_indices = test_data[test_data['capacity_Engine'].isnull()].index

    # Fill missing values with the median of the model or overall median if not available
    for idx in missing_indices:
        model_value = test_data.loc[idx, 'model']
        if model_value in model_median_dict:
            test_data.loc[idx, 'capacity_Engine'] = model_median_dict[model_value]
        else:
            test_data.loc[idx, 'capacity_Engine'] = overall_median
    test_data.loc[test_data['model'] == "אטראז'", 'capacity_Engine'] = test_data.loc[test_data['model'] == "אטראז'", 'capacity_Engine'].replace(80, 1200)

    # Fill nulls in Engine_type
    test_data['Engine_type'].fillna('בנזין', inplace=True)

    # Map locations to broader categories
    if 'Area' in test_data.columns:
        test_data['Region'] = test_data['Area'].apply(map_location_to_region)
        print("Region column added:")
        print(test_data[['Area', 'Region']].head())

    # Drop unnecessary columns if they exist
    columns_to_drop = ['Area', 'City', 'Cre_date', 'Repub_date', 'Description', 'Color', 'Description', 'Pic_num', 'Test', 'Supply_score']
    test_data = test_data.drop(columns=[col for col in columns_to_drop if col in test_data.columns])

    # Handle missing values in Km
    test_data['Km'].replace(0, np.nan, inplace=True)
    missing_indices = test_data[test_data['Km'].isnull()].index

    # Fill missing values with the mean of the year or overall mean if not available
    for idx in missing_indices:
        year_value = test_data.loc[idx, 'Year']
        if year_value in year_km_mean_dict:
            test_data.loc[idx, 'Km'] = year_km_mean_dict[year_value]
        else:
            test_data.loc[idx, 'Km'] = overall_km_mean

    # Handle outliers in test data
    for year, mean_value in year_km_mean_dict.items():
        year_group = test_data[test_data['Year'] == year]
        if not year_group.empty:
            mean = mean_value
            std = year_group['Km'].std()
            z_scores = (year_group['Km'] - mean) / std
            outliers_indices = year_group[np.abs(z_scores) > 3].index
            test_data.loc[outliers_indices, 'Km'] = mean_value

    # Drop unnecessary columns
    test_data = test_data.drop(columns=['Pic_num'], errors='ignore')

    # Clean model names
    test_data['model'] = test_data['model'].apply(clean_model)

    # Calculate model stats for capacity_Engine
    model_stats = test_data.groupby('model')['capacity_Engine'].agg(['median', 'std']).reset_index()
    model_stats.columns = ['model', 'median_capacity_Engine', 'std_capacity_Engine']

    # Replace values less than 500 with the median for the same model
    for index, row in model_stats.iterrows():
        model = row['model']
        median_value = row['median_capacity_Engine']
        test_data.loc[(test_data['model'] == model) & (test_data['capacity_Engine'] < 500), 'capacity_Engine'] = median_value

    # Handle outliers greater than 5000 based on standard deviation condition
    for index, row in model_stats.iterrows():
        model = row['model']
        median_value = row['median_capacity_Engine']
        std_value = row['std_capacity_Engine']
        if std_value > 1000:
            test_data.loc[(test_data['model'] == model) & (test_data['capacity_Engine'] > 5000), 'capacity_Engine'] = median_value

    # Handle values greater than 10000
    test_data.loc[test_data['capacity_Engine'] > 10000, 'capacity_Engine'] = test_data.loc[test_data['capacity_Engine'] > 10000, 'capacity_Engine'] / 10

    # Fill outliers less than 500 and zeros with overall median for models not present in training data
    test_data.loc[(test_data['capacity_Engine'] < 500) | (test_data['capacity_Engine'] == 0), 'capacity_Engine'] = overall_median

    # הדפסת עמודות לאימות
    print("Columns in test_data after processing:")
    print(test_data.columns)

    # Convert categorical variables into dummy/indicator variables
    test_data = pd.get_dummies(test_data, columns=['model', 'Engine_type', 'manufactor', 'Gear', 'Region'])

    # Ensure both datasets have the same columns
    test_data = test_data.reindex(columns=train_columns, fill_value=0)

    # הדפסת הנתונים המעובדים לבדיקה
    print(test_data)

    return test_data
