from transformers import TapasTokenizer, TFTapasForQuestionAnswering
import pandas as pd
import datetime


def execute_query(query, csv_file):
    a = datetime.datetime.now()

    table = pd.read_csv(csv_file.name, delimiter=",")
    table.fillna(0, inplace=True)
    table = table.astype(str)

    model_name = "google/tapas-base-finetuned-wtq"
    model = TFTapasForQuestionAnswering.from_pretrained(model_name)
    tokenizer = TapasTokenizer.from_pretrained(model_name)

    queries = [query]

    inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
    outputs = model(**inputs)

    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits, outputs.logits_aggregation
    )

    # let's print out the results:
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answers.append(table.iat[coordinates[0]])
        else:
            # multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(cell_values)

    for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
        if predicted_agg != "NONE":
            answers.append(predicted_agg)

    query_result = {
        "query": query,
        "result": answers
    }
    print(query_result)

    b = datetime.datetime.now()
    print(b - a)

    return query_result, table


