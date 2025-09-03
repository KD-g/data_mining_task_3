# import pandas as pd

# def prepare_data(df):
#     """
#     Preprocesses the input DataFrame by filling missing values and selecting features.
#     Expects columns: 'room_num', 'area', 'floor', 'total_floors', 'has_parking', 'is_furnished', and optionally 'num_of_images'.
#     """
#     df = df.copy()
    
#     # Ensure 'num_of_images' column exists (from uploaded images count)
#     if 'num_of_images' not in df.columns:
#         df['num_of_images'] = 0
    
#     # Fill missing values and convert types
#     df['has_parking'] = df['has_parking'].fillna(0).astype(int)
#     df['is_furnished'] = df['is_furnished'].fillna(0).astype(int)
#     df['room_num'] = df['room_num'].fillna(df['room_num'].median())
#     df['area'] = df['area'].fillna(df['area'].median())
#     df['floor'] = pd.to_numeric(df['floor'].str.split().str[0], errors='coerce').astype('Int64')
#     df['floor'] = df['floor'].fillna(df['floor'].median())
#     df['total_floors'] = df['total_floors'].fillna(df['total_floors'].median())
#     df['num_of_images'] = df['num_of_images'].fillna(0).astype(int)
    
#     # Select only the relevant features for the model
#     features = ['room_num', 'area', 'floor', 'total_floors', 'has_parking', 'is_furnished', 'num_of_images', 'price']
#     return df[features]


# import requests
# import pandas as pd
# import numpy as np

# def prepare_data(df):
    
#     df = df.copy()   
    
#     # Filter out invalid or outlier entries
#     if 'price' in df.columns:
#         df = df[df['price'].between(2500, 40000)]
        
#     df = df[~(df['neighborhood'].isna() & df['address'].isna())]
#     df = df.drop_duplicates(subset=['neighborhood', 'address', 'floor', 'room_num'])
#     df = df[~df['description'].astype(str).str.contains('שותפ|שותף|סאבלט|למכירה|חנות|חניה', regex=True, na=False)]
#     df = df[~df['property_type'].astype(str).str.contains('סאבלט|למכירה|מחסן|חנות|חניה', regex=True)] 
    
    
#     df.loc[df['total_floors'] >= 8, 'elevator'] = 1
#     df.address = df.address.fillna(df.neighborhood)
#     df.loc[df['room_num'].fillna(0)<1.0, 'room_num'] = (df.loc[df['room_num'].fillna(0)<1.0, 'description'].str.extract(r'(\d+(?:\.\d+)?)\s*חדרים')[0].astype(float))
#     df.loc[df['room_num'].fillna(0)<1.0, 'room_num'] = (df.loc[df['room_num'].fillna(0)<1.0, 'description'].str.extract(r'(\d+(?:\.\d+)?)\s*חד')[0].astype(float))
#     df.loc[df['area'].fillna(0)<1.0, 'area'] = (df.loc[df['area'].fillna(0)<1.0, 'description'].str.extract(r'(\d+(?:\.\d+)?)\s*מטר רבוע')[0].astype(float))
#     df.loc[df['area'].fillna(0)<1.0, 'area'] = (df.loc[df['area'].fillna(0)<1.0, 'description'].str.extract(r'(\d+(?:\.\d+)?)\s*מ"ר')[0].astype(float))
#     df.loc[df['area'].fillna(0)<1.0, 'area'] = (df.loc[df['area'].fillna(0)<1.0, 'description'].str.extract(r'(\d+(?:\.\d+)?)\s*מר')[0].astype(float))
#     df.loc[df['description'].str.contains('דירת חדר', na=False), 'room_num'] = 1.0
#     df.loc[df['description'].str.contains('דירת חדר וחצי', na=False), 'room_num'] = 1.5
#     df.loc[df['room_num'] < 1.0, 'room_num'] = np.nan
#     df.loc[df['floor'].astype(str).str.contains('ראשונה', regex=True), 'floor'] = 1.0
#     df.loc[df['floor'].astype(str).str.contains('מרתף', regex=True), 'floor'] = -1.0  
#     df.loc[df['floor'].astype(str).str.contains('קרקע', regex=True), 'floor'] = 0.0
#     df['floor'] = df['floor'].astype(str).str.extract(r'(\d+)')[0]
    
#     # Simple fills
#     df['address'] = df['address'].fillna(df['neighborhood'])
#     df['description'] = df['description'].fillna('')
#     df['garden_area'] = df['garden_area'].fillna(0)
#     df['num_of_images'] = df['num_of_images'].fillna(0)
#     df['floor'] = pd.to_numeric(df['floor'], errors='coerce')    
    
#     df.loc[df['neighborhood'] == 'לינקולן', 'neighborhood'] = 'מונטיפיורי'
#     df.loc[df['neighborhood'] == 'מע"ר צפוני', 'neighborhood'] = 'פארק צמרת'
#     socio_economic_dict = {
#         2: ["יפו ג", "יפו ד", "יפו ב", "מרכז יפו מזרחית לשדרות ירושלים", "תל כביר", "התקווה"],
#         3: ["עג'מי", "דקר", "שפירא", "קרית שלום", "עזרא", "נוה אליעזר", "לבנה"],
#         4: ["צהלון", "שיכוני חסכון", "נוה שאנן", "יד אליהו", "כפיר", "נוה ברבור", "מערב", 'אזורי חן', "נוה חן", "ניר אביב"],
#         5: ["נוה שרת", "תל חיים"],
#         6: ["רמת אביב", "הדר יוסף", "כרם התימנים", "צפון יפו", "גבעת הרצל", "פלורנטין", "רמת הטייסים"],
#         7: ["נחלת יצחק", "בצרון", "רמת ישראל", "שבזי"],
#         8: ["רמת החייל", "הצפון הישן החלק המרכזי", "הצפון הישן החלק הצפוני", "הצפון הישן החלק הדרום מזרחי", "הצפון הישן החלק הדרום מערבי", "לב תל אביב החלק הצפוני", "לב תל אביב החלק הדרומי", "בית שטראוס", "לינקולן", "מונטיפיורי"],
#         9: ["נוה אביבים", "נאות אפקה א", "נאות אפקה ב", "שיכון בבלי", "הצפון החדש החלק הצפוני", "הצפון החדש החלק הדרומי", "הצפון החדש סביבת ככר המדינה", "תכנית ל", "הגוש הגדול", "הקריה"],
#         10: ["חוף הצוק", "כוכב הצפון", "רמת אביב ג", "רמת אביב החדשה", "אפקה", "תל ברוך צפון", "מעוז אביב א", "המשתלה", "גני צהלה", "רביבים", "פארק צמרת"],
#     }
#     neighborhood_to_index = {neigh: index for index, neighs in socio_economic_dict.items() for neigh in neighs}
#     df['socio_economic'] = df['neighborhood'].map(neighborhood_to_index)
    
#     def get_distance(destination):
#         url = "https://maps.googleapis.com/maps/api/distancematrix/json"
#         params = {
#             "origins": 'דיזינגוף, תל אביב',
#             "destinations": destination,
#             "key": 'AIzaSyBK1ndFqHoEsfIwJnRhskfd1dqsFXEjw9I',
#         }
#         data = requests.get(url, params=params).json()

#         if data["status"] == "OK":
#             element = data["rows"][0]["elements"][0]
#             if element["status"] == "OK":
#                 distance_text = element["distance"]["text"] 
#                 distance_number = float(distance_text.split()[0])*1000
#                 return distance_number
    
#     # Distance from center: fill missing with median and cap outliers
#     df.loc[df['distance_from_center'].isna(), 'distance_from_center'] = (df.loc[df['distance_from_center'].isna(), 'address'].astype(str)+' תל אביב').apply(get_distance)
#     df['distance_from_center'] = pd.to_numeric(df['distance_from_center'], errors='coerce')
#     df.loc[df['distance_from_center']<15.0, 'distance_from_center'] *= 1000.0
#     df['distance_from_center'] = df['distance_from_center'].map(lambda x: f'{x:.2f}')
#     df['distance_from_center'] = pd.to_numeric(df['distance_from_center'], errors='coerce')
#     df.loc[df['distance_from_center']>20000, 'distance_from_center'] = (df.loc[df['distance_from_center']>20000, 'address'].astype(str)+' תל אביב').apply(get_distance)
    
#     # Create text-based features from description (e.g., length and word count)
#     df['description_length'] = df['description'].str.len()
#     df['description_wordcount'] = df['description'].apply(lambda x: len(x.split())) 
    
#     # Binary features: fill missing with 0 (assume absence)
#     binary_cols = ['has_parking','has_storage','elevator','ac','handicap',
#                    'has_bars','has_safe_room','has_balcony','is_furnished','is_renovated']
#     for col in binary_cols:
#         if col in df.columns:
#             df[col] = df[col].fillna(0).astype(int)    
    
#     df['floor'] = df['floor'].fillna(df['floor'].median())
#     df['num_of_payments'] = df['num_of_payments'].fillna(df['num_of_payments'].mode()[0])
#     df['days_to_enter'] = df['days_to_enter'].fillna(df['days_to_enter'].mode()[0])    
#     df['socio_economic'] = df['socio_economic'].fillna(df['socio_economic'].mean())
        
#     def fill_room_num(row):
#         candidates = df[df['room_num'].notna() & df['area'].notna()]
#         closest_idx = (candidates['area'] - row['area']).abs().idxmin()
#         return candidates.loc[closest_idx, 'room_num']
#     df.loc[df['room_num'].isna(), 'room_num'] = df[df['room_num'].isna()].apply(fill_room_num, axis=1)
#     df.loc[df['area']>=300, 'area'] = df['area'].median()        
#     df.loc[df['area'].isna(), 'area'] = df['area'].median()
#     df.loc[df['room_num'].isna(), 'room_num'] = df[df['room_num'].isna()].apply(fill_room_num, axis=1)
        
#     df['building_tax'] = df['building_tax'].fillna(df.groupby('neighborhood')['building_tax'].transform('median'))
#     df = df[df['distance_from_center']<20000]

#     # Combine or drop very rare neighborhoods
#     counts = df['neighborhood'].value_counts()
#     rare = counts[counts < 4].index
#     df.loc[df['neighborhood'].isin(rare), 'neighborhood'] = 'Other'
    
    
#     # Normalize property_type labels to general categories
#     def normalize_prop(val):
#         val = str(val)
#         if 'גג' in val or 'פנטהאוז' in val or 'דופלקס' in val:
#             return 'גג/פנטהאוז'
#         elif 'דירת גן' in val:
#             return 'דירת גן'
#         elif 'סטודיו' in val or 'לופט' in val or 'יחידת' in val:
#             return 'יחידת דיור'
#         else:
#             return 'דירה'
#     df['property_type'] = df['property_type'].apply(normalize_prop)

#     # Drop unused columns
#     df = df.drop(columns=['address','description'])
    
#     return df
    


    
import requests
import pandas as pd
import numpy as np

def prepare_data(df):
    df = df.copy()   

    # --- Filter out invalid or outlier entries ---
    if 'price' in df.columns:
        df = df[df['price'].between(2500, 40000)]

    df = df[~(df['neighborhood'].isna() & df['address'].isna())]
    df = df.drop_duplicates(subset=['neighborhood', 'address', 'floor', 'room_num'])
    df = df[~df['description'].astype(str).str.contains('שותפ|שותף|סאבלט|למכירה|חנות|חניה', regex=True, na=False)]
    df = df[~df['property_type'].astype(str).str.contains('סאבלט|למכירה|מחסן|חנות|חניה', regex=True)] 

    # --- Basic cleaning ---
    df.loc[df['total_floors'] >= 8, 'elevator'] = 1
    df['address'] = df['address'].fillna(df['neighborhood'])

    # Room and area extraction from description
    df.loc[df['room_num'].fillna(0)<1.0, 'room_num'] = (
        df.loc[df['room_num'].fillna(0)<1.0, 'description']
        .str.extract(r'(\d+(?:\.\d+)?)\s*חדרים')[0]
        .astype(float)
    )
    df.loc[df['room_num'].fillna(0)<1.0, 'room_num'] = (
        df.loc[df['room_num'].fillna(0)<1.0, 'description']
        .str.extract(r'(\d+(?:\.\d+)?)\s*חד')[0]
        .astype(float)
    )
    for pat in [r'(\d+(?:\.\d+)?)\s*מטר רבוע', r'(\d+(?:\.\d+)?)\s*מ"ר', r'(\d+(?:\.\d+)?)\s*מר']:
        df.loc[df['area'].fillna(0)<1.0, 'area'] = (
            df.loc[df['area'].fillna(0)<1.0, 'description']
            .str.extract(pat)[0]
            .astype(float)
        )

    df.loc[df['description'].str.contains('דירת חדר וחצי', na=False), 'room_num'] = 1.5
    df.loc[df['description'].str.contains('דירת חדר', na=False), 'room_num'] = 1.0
    df.loc[df['room_num'] < 1.0, 'room_num'] = np.nan

    df.loc[df['floor'].astype(str).str.contains('ראשונה', regex=True), 'floor'] = 1.0
    df.loc[df['floor'].astype(str).str.contains('מרתף', regex=True), 'floor'] = -1.0  
    df.loc[df['floor'].astype(str).str.contains('קרקע', regex=True), 'floor'] = 0.0
    df['floor'] = df['floor'].astype(str).str.extract(r'(\d+)')[0]

    # --- Fill missing values ---
    df['address'] = df['address'].fillna(df['neighborhood'])
    df['description'] = df['description'].fillna('')
    df['garden_area'] = df['garden_area'].fillna(0)
    df['num_of_images'] = df['num_of_images'].fillna(0)
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')    

    # Fix some neighborhood names
    df.loc[df['neighborhood'] == 'לינקולן', 'neighborhood'] = 'מונטיפיורי'
    df.loc[df['neighborhood'] == 'מע"ר צפוני', 'neighborhood'] = 'פארק צמרת'

    # Socio-economic mapping
    socio_economic_dict = {
        2: ["יפו ג", "יפו ד", "יפו ב", "מרכז יפו מזרחית לשדרות ירושלים", "תל כביר", "התקווה"],
        3: ["עג'מי", "דקר", "שפירא", "קרית שלום", "עזרא", "נוה אליעזר", "לבנה"],
        4: ["צהלון", "שיכוני חסכון", "נוה שאנן", "יד אליהו", "כפיר", "נוה ברבור", "מערב", 'אזורי חן', "נוה חן", "ניר אביב"],
        5: ["נוה שרת", "תל חיים"],
        6: ["רמת אביב", "הדר יוסף", "כרם התימנים", "צפון יפו", "גבעת הרצל", "פלורנטין", "רמת הטייסים"],
        7: ["נחלת יצחק", "בצרון", "רמת ישראל", "שבזי"],
        8: ["רמת החייל", "הצפון הישן החלק המרכזי", "הצפון הישן החלק הצפוני", "הצפון הישן החלק הדרום מזרחי", "הצפון הישן החלק הדרום מערבי", "לב תל אביב החלק הצפוני", "לב תל אביב החלק הדרומי", "בית שטראוס", "לינקולן", "מונטיפיורי"],
        9: ["נוה אביבים", "נאות אפקה א", "נאות אפקה ב", "שיכון בבלי", "הצפון החדש החלק הצפוני", "הצפון החדש החלק הדרומי", "הצפון החדש סביבת ככר המדינה", "תכנית ל", "הגוש הגדול", "הקריה"],
        10: ["חוף הצוק", "כוכב הצפון", "רמת אביב ג", "רמת אביב החדשה", "אפקה", "תל ברוך צפון", "מעוז אביב א", "המשתלה", "גני צהלה", "רביבים", "פארק צמרת"],
    }
    neighborhood_to_index = {neigh: idx for idx, neighs in socio_economic_dict.items() for neigh in neighs}
    df['socio_economic'] = df['neighborhood'].map(neighborhood_to_index)

    # --- Distance calculation ---
    def get_distance(destination):
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": 'דיזינגוף, תל אביב',
            "destinations": destination,
            "key": 'AIzaSyBK1ndFqHoEsfIwJnRhskfd1dqsFXEjw9I',
        }
        try:
            data = requests.get(url, params=params).json()
            if data.get("status") == "OK":
                element = data["rows"][0]["elements"][0]
                if element["status"] == "OK":
                    distance_text = element["distance"]["text"]
                    distance_number = float(distance_text.split()[0]) * 1000
                    return float(distance_number)
        except Exception:
            return 1500
        return 1500

    # Force numeric dtype
    df['distance_from_center'] = pd.to_numeric(df['distance_from_center'], errors='coerce')

    mask_na = df['distance_from_center'].isna()
    df.loc[mask_na, 'distance_from_center'] = (
        df.loc[mask_na, 'address'].astype(str) + ' תל אביב'
    ).apply(get_distance)

    df.loc[df['distance_from_center'] < 15.0, 'distance_from_center'] *= 1000.0
    df['distance_from_center'] = pd.to_numeric(df['distance_from_center'], errors='coerce')

    mask_outlier = df['distance_from_center'] > 20000
    df.loc[mask_outlier, 'distance_from_center'] = (
        df.loc[mask_outlier, 'address'].astype(str) + ' תל אביב'
    ).apply(get_distance)

    # --- Text features ---
    df['description_length'] = df['description'].str.len()
    df['description_wordcount'] = df['description'].apply(lambda x: len(x.split()))

    # --- Binary features ---
    binary_cols = ['has_parking','has_storage','elevator','ac','handicap',
                   'has_bars','has_safe_room','has_balcony','is_furnished','is_renovated']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # --- More fills ---
    df['floor'] = df['floor'].fillna(df['floor'].median())
    df['num_of_payments'] = df['num_of_payments'].fillna(df['num_of_payments'].mode()[0])
    df['days_to_enter'] = df['days_to_enter'].fillna(df['days_to_enter'].mode()[0])    
    df['socio_economic'] = df['socio_economic'].fillna(df['socio_economic'].mean())

    # --- Room num imputation ---
    def fill_room_num(row):
        candidates = df[df['room_num'].notna() & df['area'].notna()]
        diffs = (candidates['area'] - row['area']).abs()
        if diffs.notna().any():
            closest_idx = diffs.idxmin()
            return candidates.loc[closest_idx, 'room_num']
        return np.nan

    df.loc[df['room_num'].isna(), 'room_num'] = df[df['room_num'].isna()].apply(fill_room_num, axis=1)
    df.loc[df['area']>=300, 'area'] = df['area'].median()
    df.loc[df['area'].isna(), 'area'] = df['area'].median()
    df.loc[df['room_num'].isna(), 'room_num'] = df[df['room_num'].isna()].apply(fill_room_num, axis=1)

    # --- Fill building_tax ---
    df['building_tax'] = df['building_tax'].fillna(
        df.groupby('neighborhood')['building_tax'].transform('median')
    )

    # --- Filter extreme distances ---
    df = df[df['distance_from_center'] < 20000]

    # --- Rare neighborhood grouping ---
    counts = df['neighborhood'].value_counts()
    rare = counts[counts < 4].index
    df.loc[df['neighborhood'].isin(rare), 'neighborhood'] = 'Other'

    # --- Normalize property_type ---
    def normalize_prop(val):
        val = str(val)
        if 'גג' in val or 'פנטהאוז' in val or 'דופלקס' in val:
            return 'גג/פנטהאוז'
        elif 'דירת גן' in val:
            return 'דירת גן'
        elif 'סטודיו' in val or 'לופט' in val or 'יחידת' in val:
            return 'יחידת דיור'
        else:
            return 'דירה'
    df['property_type'] = df['property_type'].apply(normalize_prop)

    # --- Drop unused ---
    df = df.drop(columns=['address','description'])

    return df



#######################


train_df = pd.read_csv('train.csv')
print("raw shape:", train_df.shape)

train_df = prepare_data(train_df)

print("after prepare_data shape:", train_df.shape)
if train_df.empty:
    print("DataFrame is empty after prepare_data — aborting.")
    # Useful diagnostics:
    temp = pd.read_csv('train.csv')
    print("Counts before preprocess:")
    print(" total rows:", temp.shape[0])
    print(" price between 2500-40000:", temp['price'].between(2500,40000).sum())
    print(" rows with neighborhood or address:", (~(temp['neighborhood'].isna() & temp['address'].isna())).sum())
    print(" rows without banned description words:", (~temp['description'].astype(str).str.contains('שותפ|שותף|סאבלט|למכירה|חנות|חניה', regex=True, na=False)).sum())
    # distance diagnostics (if present)
    if 'distance_from_center' in temp.columns:
        print("distance_from_center NA count (raw):", temp['distance_from_center'].isna().sum())
    raise ValueError("Empty DataFrame after prepare_data — check filters and get_distance")
