from api.helpers.common import *
from api.helpers.config import dataset_config as cfg


def read_data_from_api(list_post_api, detail_post_api, output_dir, dataset,
                       full_data=False, s_time=None, e_time=None, type='crawl'):
    if dataset == 'MostPortal':
        success = login_mostportal()
        if not success:
            print('Error login!')
            return

    resp = requests.get(list_post_api)
    if resp.status_code != 200:
        print("error ", resp.status_code)
    else:
        if type == 'mp':
            output_path = os.path.join(output_dir, 'post_' + dataset + '_MP' + '.csv')
        elif type == 'ht':
            output_path = os.path.join(output_dir, 'post_' + dataset + '_HT' + '.csv')
        else:
            output_path = generate_output_filepath(output_dir, dataset, 'post', s_time, e_time)

        post_list = read_id_list(output_dir + '/post_' + dataset + '_full_all.csv') if full_data else []
        data = resp.json() if is_json(resp.text) else get_mostportal_resp_content(resp.text, normalize=True)
        parse_json_data(data, dataset, detail_post_api, post_list, output_dir, output_path,
                        full_data, s_time, e_time)
        message = 'Write post data done!'
        print(message)
        return message


def parse_json_data(data, dataset, detail_post_api, post_list, output_dir, output_path,
                    full_data, s_time, e_time):
    all_posts = []
    new_posts = []
    batch_size = 50
    batch_count = 0
    output_path_full = output_dir + '/post_' + dataset + '_full_all.csv'
    if dataset == 'MostPortal':
        cat_path = os.path.join(output_dir, 'cat_' + dataset + '_all' + '.csv')
        cat_dict = get_reverse_dict(cat_path)

    for item in data:
        post_id = post_title = cat_id = datetime = ''

        if dataset == 'QNPortal':
            post_id = int(item['IDBaiViet'])
            post_title = item['TieuDe']
            cat_name = item['TenChuyenMuc'].lower()
            cat_id = generate_id_from_str(cat_name)
            datetime = int(re.sub(r'\D', '', item['NgayDang']))
            datetime = round(datetime/1000)  # datetime of QN data is in millisecond format

        elif dataset == 'PTIT' or dataset == 'PTITGiaovu':
            post_id = int(item['id'])
            post_title = html.unescape(item['title'])
            cat_id = [cat['id'] for cat in item['categories']]
            cat_id = '' if len(cat_id) <= 0 else ';'.join(map(str, cat_id))

        elif dataset == 'MostPortal':
            post_id = int(item['id'])
            post_title = item['title']
            cat_name = item['catname']
            cat_id = cat_dict[cat_name] if cat_name in cat_dict else ''
            datetime = convert_datetime_mostportal(item['publisheddate'])

        if in_time_range(s_time, e_time, datetime):
            all_posts.append({
                'id': post_id,
                'title': post_title,
                'cat_id': cat_id,
                'datetime': datetime
            })

        if full_data and (post_id not in post_list):
            post_data_full = {
                'id': post_id,
                'title': post_title,
                'cat_id': cat_id,
                'datetime': datetime
            }

            post_data_full, res_code = get_post_detail(post_data_full, detail_post_api, item, dataset, output_dir)
            while res_code == -1:
                print('Retry in 5 minutes...')
                time.sleep(5*60)
                post_data_full, res_code = get_post_detail(post_data_full, detail_post_api, item, dataset, output_dir)
            new_posts.append(post_data_full)

            batch_count += 1
            if batch_count >= batch_size:
                append_list_to_csv(new_posts, output_path_full)
                new_posts.clear()

    append_list_to_csv(all_posts, output_path, mode='w')
    if len(new_posts) > 0:
        append_list_to_csv(new_posts, output_path_full)


def get_post_detail(post_data, detail_post_api, raw_item, dataset, output_dir):
    if dataset == 'QNPortal':
        print('New post:', detail_post_api + str(post_data['id']))
        resp = requests.get(detail_post_api + str(post_data['id']))
        if resp.status_code != 200:
            print("error ", resp.status_code)
            return post_data, -1
        else:
            data = resp.json()
            if len(data) > 0:
                html_content = html.unescape(data[0]['NoiDung'])
                soup = BeautifulSoup(html_content, "lxml")
                print('='*30)
                print(html_content)

                txt_content = []
                for p in soup.find_all('p'):
                    if p.text.strip() != '':
                        txt_content.append(preprocess_text(p.text))

                for tr in soup.find_all('tr'):
                    row_text = []
                    for td in tr.find_all('td'):
                        if td.text.strip() != '':
                            row_text.append(td.text.strip())
                    txt_content.append(preprocess_text('\\t'.join(row_text)))

                post_data['content'] = '\\n'.join(txt_content)
                post_data['summary'] = preprocess_text(data[0]['TomTat'])
                print(post_data['content'])

                save_attachments(soup, post_data['id'], dataset, output_dir)

                return post_data, 0

    elif dataset == 'PTIT' or dataset == 'PTITGiaovu':
        print('New post:', str(post_data['id']))
        if raw_item['content'] is None:
            raw_item['content'] = ''
        html_content = html.unescape(raw_item['content'])
        soup = BeautifulSoup(html_content, "lxml")
        print('=' * 30)
        print(html_content)

        txt_content = []
        txt_summary = ''
        for p in soup.find_all('p'):
            if not p.find('strong') and p.text.strip() != '':
                txt_content.append(preprocess_text(p.text))
            elif p.find('strong') and txt_summary == '':
                txt_summary = preprocess_text(p.text)

        for tr in soup.find_all('tr'):
            row_text = []
            for td in tr.find_all('td'):
                if td.text.strip() != '':
                    row_text.append(td.text.strip())
            txt_content.append(preprocess_text('\\t'.join(row_text)))

        post_data['content'] = '\\n'.join(txt_content)
        post_data['summary'] = txt_summary
        print(post_data['content'])

        save_attachments(soup, post_data['id'], dataset, output_dir)

        return post_data, 0

    elif dataset == 'MostPortal':
        print('New post:', detail_post_api + str(post_data['id']))
        resp = requests.get(detail_post_api + str(post_data['id']))
        if resp.status_code != 200:
            print("error ", resp.status_code)
        else:
            data = get_mostportal_resp_content(resp.text, normalize=True, type='Getpostsbyid')
            if len(data) > 0:
                html_content = html.unescape(data[0]['content'])
                soup = BeautifulSoup(html_content, "lxml")
                print('='*30)
                print(html_content)

                txt_content = []
                for p in soup.find_all('p'):
                    if p.text.strip() != '' and not p.find_parent('td'):
                        txt_content.append(preprocess_text(p.text))

                for tr in soup.find_all('tr'):
                    row_text = []
                    for td in tr.find_all('td'):
                        if td.text.strip() != '':
                            row_text.append(td.text.strip())
                    txt_content.append(preprocess_text('\\t'.join(row_text)))

                post_data['content'] = '\\n'.join(txt_content)
                post_data['summary'] = preprocess_text(html.unescape(data[0]['description']))
                print(post_data['content'])

                save_attachments(soup, post_data['id'], dataset, output_dir)

                return post_data, 0
    return None, 0


def save_attachments(soup, post_id, dataset, output_dir):
    attachments = []
    for a in soup.find_all('a', href=True):
        if re.match(r'.*?\.(docx?|xlsx?|pptx?|pdf)', a['href']):
            file_ulr = a['href']
            if not file_ulr.startswith('http'):
                file_ulr = get_base_url(dataset) + file_ulr
            if 'nafosted.vn' in file_ulr:
                file_ulr = file_ulr.replace('nafosted.vn', 'nafosted.gov.vn')
            file_name = str(post_id) + '_' + file_ulr.strip('/').split('/')[-1]
            file_path = output_dir + '/attachments/' + file_name
            if not os.path.exists(output_dir + '/attachments'):
                os.makedirs(output_dir + '/attachments')
            if not os.path.exists(file_path):
                try:
                    resp = requests.get(file_ulr)
                    if resp.status_code == 200:
                        print('New URL:', file_ulr)
                        attachments.append(file_name)
                        with open(file_path, 'wb') as f:
                            f.write(resp.content)
                except Exception as e:
                    print(e)
                    print('Error URL:', file_ulr)
                    continue
    if len(attachments) > 0:
        attachment_path = os.path.join(output_dir, 'attachment_' + dataset + '_all' + '.csv')
        write_dict_to_csv({post_id: '|'.join(attachments)}, attachment_path, mode='a')


def get_post_title_from_local(dataset, items):
    post_list = {}
    with open(os.path.join(cfg[dataset]["folder"], cfg[dataset]["post_detail_db"]), encoding="UTF8") as f:
        rd = csv.DictReader(f, delimiter=',')
        for row in rd:
            post_list[str(row["id"])] = row

    print(len(post_list))

    if isinstance(items, str):
        if items in post_list.keys():
            print(post_list[items])
            return post_list[items]["title"]
        else:
            print("key is not exist")
    elif isinstance(items, list):
        result = []
        for item in items:
            if item['id'] in post_list.keys():
                # print(id, post_list[id]['title'])
                item['title']=post_list[item['id']]['title']
                result.append(item)
            else:
                print("key "+item['id']+" is not exist")
        return result
    else:
        print("invalid id type")


def map_datetime_to_old_data(list_post_api, output_dir, dataset):
    if dataset == 'MostPortal':
        success = login_mostportal()
        if not success:
            print('Error login!')
            return

    resp = requests.get(list_post_api)
    if resp.status_code != 200:
        print("error ", resp.status_code)
    else:
        data = resp.json() if is_json(resp.text) else get_mostportal_resp_content(resp.text, normalize=True)

        datetime_dict = {}
        for item in data:
            if dataset == 'QNPortal':
                post_id = int(item['IDBaiViet'])
                datetime = int(re.sub(r'\D', '', item['NgayDang']))
                datetime_dict[post_id] = datetime

            elif dataset == 'MostPortal':
                post_id = int(item['id'])
                datetime = convert_datetime_mostportal(item['publisheddate'])
                datetime_dict[post_id] = datetime

        fi = open(output_dir + '/post_' + dataset + '_all.csv', 'r', encoding='utf-8')
        fo = csv.writer(open(output_dir + '/post_' + dataset + '_all_new.csv', "a", encoding='utf-8', newline='\n'), quoting=csv.QUOTE_NONNUMERIC)
        reader = csv.reader(fi)
        next(reader, None)  # skip the headers

        for row in reader:
            id = int(row[0])
            title = row[1]
            cat = int(row[2]) if row[2] != '' else ''
            content = row[3]
            summary = row[4]
            if datetime_dict.get(id):
                fo.writerow([id, title, cat, datetime_dict[id], content, summary])
            # fo.writerow([id, title, cat, datetime_dict[id]])

        fi.close()


if __name__ == '__main__':
    # read_data_from_api('http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/ListBaiviet',
    #                    'http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/GetbaivietTheoId?Idbaiviet=',
    #                    '../output/QNPortal', 'QNPortal', full_data=True)
    # read_data_from_api('http://portal.ptit.edu.vn/api/cat',
    #                    None,
    #                    '../output/PTIT', 'PTIT', full_data=True)
    # read_data_from_api('http://portal.ptit.edu.vn/giaovu/api/cat',
    #                    None,
    #                    '../output/PTITGiaovu', 'PTITGiaovu', full_data=True)
    # read_data_from_api('https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetListposts',
    #                    'https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/Getpostsbyid?id=',
    #                    '../output/MostPortal', 'MostPortal', full_data=True)

    get_post_title_from_local("QN_Portal", "131072")
    # map_datetime_to_old_data('http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/ListBaiviet',
    #                          '../output/QNPortal', 'QNPortal')
    # map_datetime_to_old_data('https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetListposts',
    #                          '../output/MostPortal', 'MostPortal')
