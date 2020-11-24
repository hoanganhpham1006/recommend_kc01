from api.helpers.common import *

def read_data_from_api(url, output_dir, dataset):
    if dataset == 'MostPortal':
        success = login_mostportal()
        if not success:
            print('Error login to MostPortal!')
            raise ValueError("Error login to MostPortal!")

    resp = requests.get(url)
    if resp.status_code != 200:
        print("error ", resp.status_code)
    else:
        data = resp.json() if is_json(resp.text) else get_mostportal_resp_content(resp.text)
        cat_dict = parse_json_data(data, dataset)
        cat_path = os.path.join(output_dir, 'cat_' + dataset + '_all' + '.csv')
        write_dict_to_csv(cat_dict, cat_path)
        message = 'Write category data done, file is saved at: ' + cat_path
        print(message)
        return message


def parse_json_data(data, dataset):
    cat_dict = {}
    for item in data:
        if dataset == 'QNPortal':
            cat_name = item['TenChuyenMuc'].lower()
            cat_id = generate_id_from_str(cat_name)
            if cat_id not in cat_dict:
                cat_dict[cat_id] = cat_name

        elif dataset == 'PTIT' or dataset == 'PTITGiaovu':
            cats = item['categories']
            for cat in cats:
                cat_name = cat['name']
                cat_id = cat['id']
                if cat_id not in cat_dict:
                    cat_dict[cat_id] = cat_name

        elif dataset == 'MostPortal':
            cat_name = item['name']
            cat_id = item['id']
            if cat_id not in cat_dict:
                cat_dict[cat_id] = cat_name

    return cat_dict


if __name__ == '__main__':
    # read_data_from_api('http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/ListBaiviet',
    #                    '..\\output\\QNPortal', 'QNPortal')
    # read_data_from_api('http://portal.ptit.edu.vn/api/cat',
    #                    '..\\output\\PTIT', 'PTIT')
    read_data_from_api('http://portal.ptit.edu.vn/giaovu/api/cat',
                       '..\\output\\PTITGiaovu', 'PTITGiaovu')
    # read_data_from_api('https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetCategorys',
    #                    '..\\output\\MostPortal', 'MostPortal')
