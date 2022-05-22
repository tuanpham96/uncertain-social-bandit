# Deploy `dash` app on `pythonanywhere`

Following [1] and [2] to deploy `dash` app:

- Go to `Web` tab then `Add a new web app > Flask > Python 3.X`
- Choose an appropriate folder
- Upload the zip file to that folder, go to console, unzip
- Then make virtual environment with:

``` bash
mkvirtualenv dash-sb --python=/usr/bin/python3.9
pip install -r requirements.txt
```

- Also remember to modify the `*app.py` file
    - Change `home_dir` to appropriate directory
    - And remove the last line: `app.run_server(debug=True)`

[1]: https://csyhuang.github.io/2018/06/24/set-up-dash-app-on-pythonanywhere/
[2]: https://github.com/conradho/dashingdemo
