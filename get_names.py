import pandas as pd

def get_city_names():
    cities_df = pd.read_csv('./data/cities.csv', sep = ';', encoding = 'utf8')

    datasets = { }
    big_data = [ ]

    print('Extracting city names')

    for _, row in cities_df.iterrows():
        country = row['Country name EN']
        city = row['Name']

        if not isinstance(city, str) or not isinstance(country, str):
            continue

        city = city + '<eos>'

        big_data.append(city)

        if country in datasets:
            datasets[country].append(city)
        else:
            datasets[country] = [ city ]
    
    return datasets, big_data