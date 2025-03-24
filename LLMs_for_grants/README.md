Build the image with

`sudo docker build -t <<name>>`

Run container with

`sudo docker run --rm -it -v .:/usr/src/app --network=host <<name>>`

Once inside the container, to use OpenAI's ChatGPT API, run

`az login`

Once logged in, run the Gradio app with

`python app.py`

To change Gradio's default port, add `server_port=<new port>` when in the `launch()` method of the Gradio interface.