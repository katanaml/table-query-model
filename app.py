from app.tapas import execute_query
import gradio as gr


def main():
    description = "Table query demo app, it runs TAPAS model. You can ask a question about tabular data, TAPAS model " \
                  "will produce the result. Think about it as SQL query running against DB table. The advantage of " \
                  "TAPAS model - there is no need to upload data to DB or process it in a spreadsheet, data can be " \
                  "processed in memory by ML model. Pre-trained TAPAS model runs on max 64 rows and 32 columns data. " \
                  "Make sure CSV file data doesn't exceed these dimensions."

    article = "<p style='text-align: center'><a href='https://katanaml.io' target='_blank'>Katana ML</a> | <a href='https://github.com/katanaml/table-query-model' target='_blank'>Github Repo</a> | <a href='https://huggingface.co/google/tapas-base-finetuned-wtq' target='_blank'>TAPAS Model</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=abaranovskij_tablequery' alt='visitor badge'></center>"

    iface = gr.Interface(fn=execute_query,
                         inputs=[gr.Textbox(label="Search query"),
                                 gr.File(label="CSV file")],
                         outputs=[gr.JSON(label="Result"),
                                  gr.Dataframe(label="All data")],
                         examples=[
                             ["What are the items with total higher than 8?", "taxables.csv"],
                             ["What is the cost for Maxwell item?", "taxables.csv"],
                             ["Show items with cost lower than 2 and tax higher than 0.05", "taxables.csv"]
                         ],
                         title="Table Question Answering (TAPAS)",
                         description=description,
                         article=article,
                         allow_flagging='never')
    # Use this config when running on Docker
    iface.launch(server_name="0.0.0.0", server_port=7000)
    # iface.launch(enable_queue=True)


if __name__ == "__main__":
    main()
