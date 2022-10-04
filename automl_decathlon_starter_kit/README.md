AutoML Decathlon starter kit
======================================

## Download public datasets
The 10 developoment datasets were hosted on Google cloud storage. First, you need to install the
[gcloud CLI](https://cloud.google.com/sdk/docs/install). Then run the following command line to download the datasets,
is:
```bash
gsutil cp -r gs://decathlon_public_datasets/dev_public <path_public_data>
```
Note that this can take some time, depending on your connection.

## Local development and testing
To make your own submission, you need to modify the
file `model.py` in `Decathlon_sample_code_submission/`, which implements the logic
of your algorithm. You can then test it on your local computer using Docker,
in the exact same environment as on the CodaLab challenge platform. 

If you are new to docker, install docker from https://docs.docker.com/get-started/. Also make sure you have
[installed nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Then at the shell, run:
```
cd path/to/decathlon_starter_kit
docker pull automldec/decathlon
docker run --gpus all --rm -it -v "$(pwd):/app/codalab" -v "<path_public_data>:/app/codalab/dev_public" -w "/app/codalab" -p 8888:8888 automldec/decathlon:latest
```
The option `-p 8888:8888` is useful for running a Jupyter notebook tutorial
inside Docker. If the port 8888 is occupied, you can use other ports, e.g. 8899, and use instead the option `-p 8899:8888`.

You will then be able to run the `ingestion program` (to produce predictions)
and the `scoring program` (to evaluate your predictions) on public data

## Run the tutorial
We provide a tutorial in the form of a Jupyter notebook. When you are in your
docker container, enter:
```
jupyter-notebook --ip=0.0.0.0 --allow-root &
```
Then copy and paste the URL containing your token. It should look like something
like that:
```
http://0.0.0.0:8888/?token=76a9ef49ecb17899f3fe290f20c5902a90973aab15be91dc
```
and select the Jupyter notebook in the menu.

## Run local test
We provide a Python script to simulate this CodaLab workflow:
```
python run_local_test.py --code_dir=./sample_code_submission --dataset_dir=./dev_public --time_budget=60
```

## Understand how a submission is evaluated

You can refer to the source code at
- Ingestion Program: `ingestion/ingestion.py`
- Scoring Program: `scoring/score.py`

The ingestion program can be run using the following command:
```bash
python ingestion/ingestion.py --dataset_dir=./dev_public --code_dir=./sample_code_submission --time_budget=60.0
```
and the scoring program can be run as follows:
```bash
python scoring/score.py --dataset_dir=./dev_public
```


## Prepare a ZIP file for submission on CodaLab
Make sure you include at least `model.py` and a `metadata` file in the submission folder. You can add an optional
`tasks_to_run.yaml` to include the tasks for the submission to run. Then zip the contents of `sample_code_submission`,
```
cd sample_code_submission/
zip -r mysubmission.zip *
```
then use the "Upload a Submission" button to make a submission to the
competition page on CodaLab platform.

Tip: to look at what's in your submission zip file without unzipping it, you
can do
```
unzip -l mysubmission.zip
```

## Acknowledgment

Some of the codes in ingestion/scoring programs and `model.py` were adapted from [past AutoDL competitions](https://github.com/zhengying-liu/autodl_starting_kit_stable).

## Contact us
If you have any questions, please contact us via:
<automl.decathlon@gmail.com>
