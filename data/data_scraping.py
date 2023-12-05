#  data scraping
import os
import requests
import pandas as pd


def download_files(url, id):
    currentPath = os.path.dirname(__file__)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filepath = os.path.join(currentPath, data, f'{id}.pdf')
        with open(filepath, 'wb') as pdf_object:
            pdf_object.write(response.content)
            print(f'{id} was successfully saved!')
            return True
    else:
        print(f'Uh oh! Could not download {id},')
        print(f'HTTP response status code: {response.status_code}')
        return False
    

if __name__ == '__main__':
    parspec_data = ['train_data','test_data']
    currentPath = os.path.dirname(__file__)

    for data in parspec_data:
        df = pd.read_csv(currentPath + f'/parspec_{data}.csv')
        for i in range(len(df)):
            e = df.iloc[i]
            # print(e['URL'])
            try:
                download_files(e['URL'], e["ID"])
            except:
                continue
        break
    print('Scraping completed')

