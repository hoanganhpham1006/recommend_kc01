import csv
import os
import requests
import json

import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
from api.helpers.config import host_config, dataset_config

def readAllDataFromApi(dataset, s_time=None, e_time=None):
    query = None
    if (s_time is not None):
        query = '{"ts":{"$gte":'+str(s_time)
        if(e_time is not None):
            query += ',"$lte":'+str(e_time)
        query += '}}'

    if query is not None:
        query = '&filter='+query
    else:
        query = ''

    url = host_config["vsmarty"]["host"]
    api_key = host_config["vsmarty"]["api_key"]

    if dataset == "PTIT" or dataset == "PTITGiaovu":
        url = host_config["207"]["host"]
        api_key = host_config["207"]["api_key"]

    if(dataset not in dataset_config):
        print("Invalid dataset! Usage -h parameter for more detail!")
        return
    cfg = dataset_config[dataset]

    resp = requests.get(url + '?api_key=' + api_key + '&dbs=' + cfg["db_name"] + '&collection='+ cfg["collection_name"] + query)
    if resp.status_code != 200:
        # This means something went wrong.
        print("error ",resp.status_code)
    else:
        total = resp.json()['total']
        print('collecton size: ',total)

        if total > 20 :
            print(url + '?api_key=' + api_key + '&dbs=' + cfg["db_name"] + '&collection=' + cfg["collection_name"] + '&limit=' + str(
                total) + query)
            resp2 = requests.get(
                url + '?api_key=' + api_key + '&dbs=' + cfg["db_name"] + '&collection=' + cfg["collection_name"] + '&limit=' + str(
                    total) + query)
            data = resp2.json()

            filename = 'user_'+cfg["dataset"]
            if s_time is not None:
                filename += '_from_'+ str(s_time)
                if e_time is not None: filename += '_to_' + str(e_time)
            else :  filename += '_all'

            # with open(os.path.join(outputDir, 'json_'+filename+'.json'), 'w') as outfile:
            #     json.dump(resp2.json(), outfile)

            events = parseJsonData(data)
            users = filterUser(events)

            print(type(events[0]))

            fileLoc = os.path.join(cfg["folder"], filename+'.csv')
            print(fileLoc)

            with open(fileLoc, 'w', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(users)



def parseJsonData(json):
    data = []
    pc_type = ['Windows', 'Mac OSX', 'Linux', 'Fedora', 'Ubuntu']
    mobile_type = ['Android', 'iOS', 'Chrome OS', 'Windows Phone', 'BlackBerry OS', 'Tizen', 'Kindle', 'Firefox OS']
    d_type_list = []
    d_id_list= []
    for item in json['collections']:
        d_id = item['d_id']
        d_pla = item['d_pla']
        d_pla_v = item['d_pla_v']
        cty = item['cty']

        if d_pla in pc_type: d_type = "pc"
        elif d_pla in mobile_type: d_type = "mobile"
        else: d_type = "Other"

        if d_pla not in d_type_list: d_type_list.append(d_pla)
        if d_id not in d_id_list: d_id_list.append(d_id)

        event = [d_id, d_type,d_pla,d_pla_v,cty]
        data.append(event)

    print('user: ',len(data))
    print(d_type_list)
    print(len(d_id_list))
    return data

def filterUser(events, threshold='10'):
    user_list = []
    tmp_list = []
    for event in events:
        tmp = ','.join(str(v) for v in event[:3])
        if tmp not in tmp_list:
            tmp_list.append(tmp)
            user_list.append(event[:4])

    print(len(user_list))
    print(len(tmp_list))
    return user_list

def main():
    # readAllDataFromApi('QN_DVC', None, None)
    # readAllDataFromApi('QN_Portal', None, None)
    readAllDataFromApi('Most_DVC', None, None)
    readAllDataFromApi('Most_Portal', None, None)
    readAllDataFromApi('PTIT', None, None)

if __name__ == '__main__':
    main()