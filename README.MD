# Table Query with Hugging Face TAPAS &#129303;

## Description

Hugging Face TAPAS model for answering questions about tabular data.

## Instructions

1. Install libraries

```
pip install -r requirements.txt
```

2. Run

```
python app.py
```

## Build and run container

1. Build container (uncomment launch call in app.py)

```
docker build --tag katanaml/table-query .
```

2. Run container

```
docker run -it -d --name table-query -p 7000:7000  katanaml/table-query:latest
```

## Structure

```
.
├── app
│   └── tapas.py
├── app.py
├── Dockerfile
├── LICENSE
├── README.md
├── taxables.csv
└── requirements.txt
```

## Author

[Katana ML](https://katanaml.io), [Andrej Baranovskij](https://github.com/abaranovskis-redsamurai)

## License

Licensed under the Apache License, Version 2.0. Copyright 2020-2021 Katana ML, Andrej Baranovskij. [Copy of the license](https://github.com/katanaml/table-query-model/blob/main/LICENSE).
