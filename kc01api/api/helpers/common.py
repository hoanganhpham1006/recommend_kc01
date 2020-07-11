import hashlib
import requests
import json
import csv
import re
import os
import time
import html

from xml.etree import ElementTree
from bs4 import BeautifulSoup
csv.field_size_limit(100000000)

MOSTPORTAL_USERNAME = 'portal'
MOSTPORTAL_PASSWORD = '123456a@'


def generate_id_from_str(s):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8


def write_dict_to_csv(dict, output_path, mode='w'):
    f = csv.writer(open(output_path, mode, encoding='utf-8', newline='\n'), quoting=csv.QUOTE_NONNUMERIC)

    for key in dict:
        f.writerow([key, dict[key]])


def append_list_to_csv(_list, output_path, mode='a'):
    f = csv.writer(open(output_path, mode, encoding='utf-8', newline='\n'), quoting=csv.QUOTE_NONNUMERIC)

    for item in _list:
        f.writerow([item[att] for att in item])


def write_list_to_csv(_list, output_path, delimiter='|'):
    f = csv.writer(open(output_path, "w", encoding='utf-8', newline='\n'),
                   quoting=csv.QUOTE_NONNUMERIC, delimiter=delimiter)

    for item in _list:
        f.writerow([att for att in item])


def is_json(text):
    try:
        json.loads(text)
    except ValueError as e:
        return False
    return True


def preprocess_text(text):
    text = text.strip()
    text = re.sub(r'[\n\r\t]+', r'\\n', text)
    return text


def generate_output_filepath(output_dir, dataset, type, s_time, e_time):
    filename = type + '_' + dataset
    if s_time is not None:
        filename += '_from_' + str(s_time)
        if e_time is not None:
            filename += '_to_' + str(e_time)
    else:
        filename += '_all'

    return os.path.join(output_dir, filename + '.csv')


def get_dict(path):
    _dict = {}
    with open(path, 'a+', encoding='utf-8') as csvfile:
        csvfile.seek(0)
        file = csv.reader(csvfile, delimiter=',')
        for row in file:
            try:
                _dict[int(row[0])] = row[1]
            except ValueError:
                continue
    return _dict


def get_reverse_dict(path):
    reverse_dict = {}
    with open(path, 'a+', encoding='utf-8') as csvfile:
        csvfile.seek(0)
        file = csv.reader(csvfile, delimiter=',')
        for row in file:
            reverse_dict[row[1]] = int(row[0])
    return reverse_dict


def read_id_list(path):
    if not os.path.exists(path):
        f_post = open(path, 'w+', encoding='utf-8')
        f_post.close()
        return []

    f_post = open(path, 'r', encoding='utf-8')
    reader = csv.reader(f_post)
    next(reader, None)  # skip the headers
    id_list = [int(rows[0]) for rows in reader]
    f_post.close()

    return id_list


def in_time_range(s_time, e_time, time):
    if s_time is None and e_time is None:
        return True
    if s_time is not None and e_time is None:
        if s_time <= time:
            return True
    if s_time is None and e_time is not None:
        if time <= e_time:
            return True
    if s_time is not None and e_time is not None:
        if s_time <= time <= e_time:
            return True
    return False


def get_base_url(dataset):
    if dataset == 'QNPortal':
        return 'http://quangnam.gov.vn/'
    if dataset == 'PTIT' or dataset == 'PTITGiaovu':
        return 'http://portal.ptit.edu.vn/'
    if dataset == 'MostPortal':
        return 'https://www.most.gov.vn/'


def get_post_id_from_url_ptit(slug, dataset):
    if dataset == 'PTITGiaovu':
        slug = 'giaovu/' + slug
    resp = requests.head('http://portal.ptit.edu.vn/' + slug, allow_redirects=True)
    if resp.status_code != 200:
        print("error ", resp.status_code)
        return None
    else:
        headers = resp.headers
        if 'Link' in headers:
            url_match = re.search(r'.*?\?p=(\d+)', headers['Link'])
            if url_match:
                return int(url_match.group(1))
    return None


def login_mostportal():
    resp = requests.post('https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/LogIn',
                         data={'strUse': MOSTPORTAL_USERNAME, 'strPass': MOSTPORTAL_PASSWORD})
    if resp.status_code != 200:
        print("error ", resp.status_code)
        return False
    else:
        root = ElementTree.fromstring(resp.content)
        if root.text == 'true':
            return True
        return False


def get_mostportal_resp_content(xml_text, normalize=False, type='GetListposts'):
    root = ElementTree.fromstring(xml_text)
    if normalize:
        return normalize_mostportal_content(root.text, type)
    return json.loads(root.text, strict=False)


def normalize_mostportal_content(json_text, type):
    # json_file = open('../data/csv/MostPortal/listposts.json', encoding='utf-8')
    # data = json_file.read().lstrip('[{').rstrip('}]')
    data = json_text.lstrip('[{').rstrip('}]')
    data = data.split('},{')

    result = []
    for item in data:
        new_item = {}
        if type == 'GetListposts':
            item = re.sub(r'[\r\n]+', ' ', item)
            id_match = re.search(r'"id":(.+),"slug"', item)
            slug_match = re.search(r'"slug":"(.+)","title"', item)
            title_match = re.search(r'"title":"(.+)","catname"', item)
            cat_match = re.search(r'"catname":"(.+)","publisheddate"', item)
            datetime_match = re.search(r'"publisheddate":"(.+)"$', item)

            new_item = {
                'id': int(id_match.group(1)) if id_match else '',
                'slug': slug_match.group(1) if slug_match else '',
                'title': title_match.group(1) if title_match else '',
                'catname': cat_match.group(1) if cat_match else '',
                'publisheddate': datetime_match.group(1) if datetime_match else ''
            }
        elif type == 'Getpostsbyid':
            item = re.sub(r'[\n\r]+', r'\t', item)
            cat_match = re.search(r'"catname":(.+),"title"', item)
            title_match = re.search(r'"title":"(.+)","description"', item)
            des_match = re.search(r'"description":"(.+)","content"', item)
            content_match = re.search(r'"content":"(.+)"$', item)

            new_item = {
                'catname': cat_match.group(1) if cat_match else '',
                'title': title_match.group(1) if title_match else '',
                'description': des_match.group(1) if des_match else '',
                'content': content_match.group(1) if content_match else ''
            }

        result.append(new_item)

    return result


def convert_datetime_mostportal(datetime_str):
    # datetime_str = '03/07/2020 15:27'
    t = time.mktime(datetime.strptime(datetime_str, "%d/%m/%Y %H:%M").timetuple())
    return int(t)
