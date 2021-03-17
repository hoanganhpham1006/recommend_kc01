import sys

from pathlib import Path
from config import host_config
from common import *

sys.path.append(str(Path(os.getcwd()).parent))


def read_data_from_api(db_name, collection, list_post_api, s_time=None, e_time=None, output_dir=os.getcwd(), dataset='None', type='crawl'):
    if dataset == 'MostPortal':
        success = login_mostportal()
        if not success:
            print('Error login to MostPortal!')
            raise ValueError("Error login to MostPortal!")

    query = None
    if s_time is not None:
        query = '{"ts":{"$gte":' + str(s_time)
        if e_time is not None:
            query += ',"$lte":' + str(e_time)
        query += '}}'

    if query is not None:
        query = '&filter=' + query
    else:
        query = ''

    url = host_config["vsmarty"]["host"]
    api_key = host_config["vsmarty"]["api_key"]

    if dataset == "PTIT" or dataset == "PTITGiaovu":
        url = host_config["207"]["host"]
        api_key = host_config["207"]["api_key"]

    print(url, api_key, sep='-')

    resp = requests.get(url + '?api_key=' + api_key + '&dbs=' + db_name + '&collection=' + collection + query)
    print(url + '?api_key=' + api_key + '&dbs=' + db_name + '&collection=' + collection + query)
    if resp.status_code != 200:
        print("error ", resp.status_code)
        print(url + '?api_key=' + api_key + '&dbs=' + db_name + '&collection=' + collection + query)
        raise ValueError("Error when retrieve transaction database!")
    else:
        total = resp.json()['total']
        print('collection size: ', total)
        if total > 20:
            print(url + '?api_key=' + api_key + '&dbs=' + db_name + '&collection=' + collection + '&limit=' + str(total) + query)
            resp2 = requests.get(url + '?api_key=' + api_key + '&dbs=' + db_name + '&collection=' + collection + '&limit=' + str(total) + query)
            data = resp2.json()
            transaction_urls = set([item['segmentation_name'].rstrip('/') for item in data['collections']
                                    if item['segmentation_name'] is not None and item['segmentation_name'] != ''])
            print('Fetch data done!')

            if type == 'mp':
                cat_path = os.path.join(output_dir, 'cat_' + dataset + '_all' + '.csv')
                url_path = os.path.join(output_dir, 'url_' + dataset + '_MP' + '.csv')
                file_path = os.path.join(output_dir, 'transaction_' + dataset + '_MP' + '.csv')
            elif type == 'ht':
                cat_path = os.path.join(output_dir, 'cat_' + dataset + '_all' + '.csv')
                url_path = os.path.join(output_dir, 'url_' + dataset + '_HT' + '.csv')
                file_path = os.path.join(output_dir, 'transaction_' + dataset + '_HT' + '.csv')
            else:
                cat_path = os.path.join(output_dir, 'cat_' + dataset + '_all' + '.csv')
                url_path = generate_output_filepath(output_dir, dataset, 'url', s_time, e_time)
                file_path = generate_output_filepath(output_dir, dataset, 'transaction', s_time, e_time)
            print('Generate url filepath done!')

            cat_dict = get_reverse_dict(cat_path)
            print('Read cat dict done!')

            append_url_dict = type != 'mp' and type != 'ht'
            url_dict = get_reverse_dict(url_path) if append_url_dict else {}
            print('Read url dict done!')

            post_dict, url_updated = get_post_dict_and_update_url(list_post_api, url_dict, url_path,
                                                                  transaction_urls, append_url_dict,
                                                                  dataset)
            print('Read post dict and update url done!')

            if url_updated:
                url_dict = get_reverse_dict(url_path)
                print('Read updated url dict done!')

            events = parse_json_data(data, cat_dict, url_dict, post_dict, dataset)
            print('Parse json data done!')

            with open(file_path, 'w', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(events)
            message = 'Write transaction data done, file is saved at: ' + file_path
            print(message)
            message2 = 'Write url data done, file is saved at: ' + url_path
            return [message, message2]
        else:
            raise ValueError("Collection size is lower than 20!")


def parse_json_data(json, cat_dict, url_dict, post_dict, dataset):
    data = []
    for item in json['collections']:
        ts = item['ts']
        d_id = item['d_id']
        url = item['segmentation_name']

        # Filter urls to keep only post's urls
        if url is not None and url != '':
            if dataset == 'QNPortal':
                url_match = re.match(r'.*?IDBaiViet=(\d+)', url)
                if url_match:
                    post_id = int(url_match.group(1))
                    cat_id = ''
                    if post_id in post_dict:
                        cat = post_dict[post_id]
                        if cat in cat_dict:
                            cat_id = cat_dict[cat]

                    event = [d_id, post_id, cat_id, ts]
                    data.append(event)

            elif dataset == 'PTIT' or dataset == 'PTITGiaovu':
                if dataset == 'PTIT':
                    pattern = r'^(?!^/$|(^/(category|giaovu|page)(/.*)?$)).*'
                else:
                    pattern = r'^(?=^/giaovu/.+$)(?!^/giaovu/category(/.*)?$).*'

                url_match = re.match(pattern, url)
                if url_match:
                    slug = url.strip('/').split('/')[-1]
                    url = ('/' + slug if dataset == 'PTIT' else '/giaovu/' + slug).lower()
                    if url in url_dict:
                        post_id = url_dict[url]
                        cat_ids = []
                        if post_id in post_dict:
                            cats = post_dict[post_id]
                            for cat in cats:
                                if cat in cat_dict:
                                    cat_ids.append(cat_dict[cat])
                        cat_ids = '' if len(cat_ids) <= 0 else ';'.join(map(str, cat_ids))

                        event = [d_id, post_id, cat_ids, ts]
                        data.append(event)

            elif dataset == 'MostPortal':
                url_match = re.match(r'.*?/vn/tin-tuc/(\d+)', url)
                if url_match:
                    post_id = int(url_match.group(1))
                    cat_id = ''
                    if post_id in post_dict:
                        cat = post_dict[post_id]
                        if cat in cat_dict:
                            cat_id = cat_dict[cat]

                    event = [d_id, post_id, cat_id, ts]
                    data.append(event)

    print('events: ', len(data))
    return data


def get_post_dict_and_update_url(list_post_api, url_dict, url_path, transaction_urls, append_url_dict, dataset):
    url_updated = False
    resp = requests.get(list_post_api)
    if resp.status_code != 200:
        print("error ", resp.status_code)
        return None
    else:
        data = resp.json() if is_json(resp.text) else get_mostportal_resp_content(resp.text, normalize=True)
        post_dict = {}
        new_url_dict = {}
        for post in data:
            if dataset == 'QNPortal':
                post_id = post['IDBaiViet']
                url = '/CMSPages/BaiViet/Default.aspx?IDBaiViet=' + str(post_id)
                if url in transaction_urls:
                    if url not in url_dict:
                        new_url_dict[post_id] = url
                    cat = post['TenChuyenMuc'].lower()
                    if post_id not in post_dict:
                        post_dict[post_id] = cat

            elif dataset == 'PTIT' or dataset == 'PTITGiaovu':
                url = '/' + post['slug'] if dataset == 'PTIT' else '/giaovu/' + post['slug']
                if url in transaction_urls:
                    if url not in url_dict:
                        post_id = get_post_id_from_url_ptit(post['slug'], dataset)
                        if post_id is not None:
                            new_url_dict[post_id] = url
                    else:
                        post_id = url_dict[url]

                    cats = post['categories']
                    cats = [c['name'] for c in cats]
                    if post_id is not None and post_id not in post_dict:
                        post_dict[post_id] = cats

            elif dataset == 'MostPortal':
                url_match = re.match(r'.*?/vn/tin-tuc/(\d+)', post['slug'])
                if url_match:
                    post_id = int(url_match.group(1))
                    url = post['slug'].replace('https://www.most.gov.vn', '').replace('http://www.most.gov.vn', '')
                    if url in transaction_urls:
                        if url not in url_dict:
                            new_url_dict[post_id] = url
                        cat = post['catname']
                        if post_id not in post_dict:
                            post_dict[post_id] = cat

        if len(new_url_dict) > 0:
            mode = 'a' if append_url_dict else 'w'
            write_dict_to_csv(new_url_dict, url_path, mode=mode)
            url_updated = True

        return post_dict, url_updated


if __name__ == '__main__':
    read_data_from_api('countly', 'did_url_r1_5d1abdb4a3d66b55d6ed38e8',
                       'http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/ListBaiviet',
                       1578926621, None, '..\\output\\QNPortal', 'QNPortal')
    # read_data_from_api('countly', 'did_url_r1_5c2c39560e66a35b7e802993',
    #                    'http://portal.ptit.edu.vn/api/cat',
    #                    1576368000, None, '..\\output\\PTIT', 'PTIT')
    # read_data_from_api('countly', 'did_url_r1_5c2c39560e66a35b7e802993',
    #                    'http://portal.ptit.edu.vn/giaovu/api/cat',
    #                    1576368000, None, '..\\output\\PTITGiaovu', 'PTITGiaovu')
    # read_data_from_api('countly', 'did_url_r1_5d49b5a011dd440567e158c2',
    #                    'https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetListposts',
    #                    1576368000, None, '..\\output\\MostPortal', 'MostPortal')
