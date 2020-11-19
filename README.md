# KC01_SESSIONBASED_APIs

Version: v1.1

Release date: Oct 29, 2020

KC01_SESSIONBASED_APIs which includes deep learning side and backend code. It supports data crawling, data preprocessing, session-based model training, model setting, and model inference as RESTFUL API. It uses django framework for backend code.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r kc01api/requirements.txt
```

## Starting system
Using gunicorn, CUDA is needed

```
rm kc01api/api/logs/*train*
CUDA_VISIBLE_DEVICES=1 gunicorn django_api_base.wsgi -b 0.0.0.0:5009 -w 1
```

## Client API
Please refer to [this POSTMAN generated document](https://www.getpostman.com/collections/57b47df7ee39c6fa6895)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
© Machine Learning and Application Lab \
Posts and Telecommunications Institute of Technology
