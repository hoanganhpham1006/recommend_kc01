from django.conf import settings

host_config = {
    "vsmarty": {
        "host": "https://tracker-29.logger.vsmarty.vn/o/db",
        "api_key": "904544e19c8d9c141fe2860dfa7aad42"
    },
    "207": {
        "host": "http://207.148.79.97/o/db",
        "api_key": "76e0e145aa7343c0b23ac8a1156a0e7e"
    }
}

dataset_config = {
    "QN_DVC": {
        "dataset": "QN",
        "host": "vsmarty",
        "db_name": "countly",
        "collection_name": "did_url_r1_5d125bae529d875d43675518",
        "folder": "D:\Working\PythonProjects\Recommendation\output\QN",
        "transaction_ht_map": "transaction_QN_HT.csv",
        "transaction_mp_map": "transaction_QN_MP.csv",
        "url_ht_map": "url_QN_HT.csv",
        "url_mp_map": "url_QN_MP.csv",
        "post_detail_db": "post_QN_all.csv",
        "list_post_api": "",
        "list_cat_api": "",
        "corpus": ""
    },
    "QN_Portal": {
        "dataset": "QNPortal",
        "host": "vsmarty",
        "db_name": "countly",
        "collection_name": "did_url_r1_5d1abdb4a3d66b55d6ed38e8",
        "folder": "D:\Working\PythonProjects\Recommendation\output\QNPortal",
        "transaction_ht_map": "transaction_QNPortal_HT.csv",
        "transaction_mp_map": "transaction_QNPortal_MP.csv",
        "transaction_all_map": "transaction_QNPortal_all.csv",
        "url_ht_map": "url_QNPortal_HT.csv",
        "url_mp_map": "url_QNPortal_MP.csv",
        "post_detail_db": "post_QNPortal_all.csv",
        "list_post_api": "http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/ListBaiviet",
        "list_cat_api": "http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/ListBaiviet",
        "corpus": "post_tf-idf_QNPortal.json"
    },
    "Most_DVC": {
        "dataset": "MostDichvucong",
        "host": "vsmarty",
        "db_name": "countly",
        "collection_name": "did_url_r1_5d49b61811dd440567e158c4",
        "folder": "D:\Working\PythonProjects\Recommendation\output\MostDichvucong",
        "transaction_ht_map": "transaction_MostDichvucong_HT.csv",
        "transaction_mp_map": "transaction_MostDichvucong_MP.csv",
        "url_ht_map": "url_MostDichvucong_HT.csv",
        "url_mp_map": "url_MostDichvucong_MP.csv",
        "post_detail_db": "post_MostDichvucong_all.csv",
        "list_post_api": "https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetListposts",
        "list_cat_api": "",
        "corpus": ""
    },
    "Most_Portal": {
        "dataset": "MostPortal",
        "host": "vsmarty",
        "db_name": "countly",
        "collection_name": "did_url_r1_5d49b5a011dd440567e158c2",
        "folder": settings.BASE_DIR + "/api/databases/MostPortal",
        "transaction_ht_map": "transaction_MostPortal_HT.csv",
        "transaction_mp_map": "transaction_MostPortal_MP.csv",
        "transaction_all_map": "transaction_MostPortal_MP.csv",
        "url_ht_map": "url_MostPortal_HT.csv",
        "url_mp_map": "url_MostPortal_MP.csv",
        "post_detail_db": "post_MostPortal_all.csv",
        "list_post_api": "https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetListposts",
        "list_cat_api": "https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetCategorys",
        "corpus": "post_tf-idf_MostPortal.json.json",
        "lda_model": "MostPortal_gensim.model"
    },
    "PTIT": {
        "dataset": "PTIT",
        "host": "207",
        "db_name": "countly",
        "collection_name": "did_url_r1_5c2c39560e66a35b7e802993",
        "folder": "D:\Working\PythonProjects\Recommendation\output\PTIT",
        "transaction_ht_map": "transaction_PTIT_HT.csv",
        "transaction_mp_map": "transaction_PTIT_MP.csv",
        "url_ht_map": "url_PTIT_HT.csv",
        "url_mp_map": "url_PTIT_MP.csv",
        "post_detail_db": "post_PTIT_all.csv",
        "list_post_api": "http://portal.ptit.edu.vn/api/cat",
        "list_cat_api": "http://portal.ptit.edu.vn/api/cat",
        "corpus": ""
    },
    "PTIT_Giaovu": {
        "dataset": "PTITGiaovu",
        "host": "207",
        "db_name": "countly",
        "collection_name": "did_url_r1_5c2c39560e66a35b7e802993",
        "folder": "D:\Working\PythonProjects\Recommendation\output\PTITGiaovu",
        "transaction_ht_map": "transaction_PTITGiaovu_HT.csv",
        "transaction_mp_map": "transaction_PTITGiaovu_MP.csv",
        "url_ht_map": "url_PTITGiaovu_HT.csv",
        "url_mp_map": "url_PTITGiaovu_MP.csv",
        "post_detail_db": "post_PTITGiaovu_all.csv",
        "list_post_api": "http://portal.ptit.edu.vn/giaovu/api/cat",
        "list_cat_api": "http://portal.ptit.edu.vn/giaovu/api/cat",
        "corpus": ""
    }
}
