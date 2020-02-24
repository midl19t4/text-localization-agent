# text-localization-agent

The code to train the agent.

## Prerequisites

You need Python 3 (preferably 3.6) installed, as well as the requirements from `requirements.txt`:

```bash
$ pip install -r requirements.txt 
```

Furthermore, you need to install the [text-localization-environment](https://github.com/midl19t4/text-localization-agent) by following its **Installation** instructions.

## Usage

Training an agent requires two files:

1. A json containing an array of image parameters:
    1. a relative path to the image file (`file_name`)
    2. an array of bounding boxes (`bounding_boxes`) in this format:  
   [x<sub>topleft</sub>, y<sub>topleft</sub>, x<sub>bottomright</sub>, y<sub>bottomright</sub>]

Overview of executable python scripts:
|File|Purpose|Example|
|---|---|---|
|`train_agent.py`| Train the agent| `python3 train_agent.py --config example.ini` |
|`test_agent.py`| Test the agent with the first `n` images (the evaluation data is saved under `<resultdir_path>/evaluation_data`)| `python3 test_agent.py 1000 --config example.ini` |
|`plot_agent.py`| Creates plots of the agents performance based on the output of `test_agent.py` (this separation is done to allow for additional evaluation of the raw testing data after training, the plots are saved under `<resultdir_path>/plots`)| `python3 plot_agent.py --config example.ini` |

## TensorBoard

If you would like the program to generate log-files appropriate for visualization in TensorBoard, you need to:

1. Install **tensorflow**
    ```bash
    $ pip install tensorflow
    ```
    (If you use Python 3.7 and the installation fails, use: `pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
    ` instead. [See here, why.](https://github.com/tensorflow/tensorflow/issues/20444#issuecomment-442767411))
1. Run the *text-localization-agent* program with  `use_tensorboard = True` in the config
   ```bash
   $ python train-agent.py --config example.ini
   ``` 
1. Start TensorBoard pointing to the `tensorboard/` directory inside the *text-localization-agent* project
   ```bash
   $ tensorboard --logdir=<path_to_text-localization-agent>/tensorboard/
   …
   TensorBoard 1.12.0 at <link to TensorBoard UI> (Press CTRL+C to quit)
   ``` 
1. Open the TensorBoard UI via the link that is provided when the `tensorboard` program is started  (http://<ip_of_the_server>:6006)

## Training on the chair's servers

To run the training on one of the chair's servers you need to:

* Clone the necessary repositories
* Create a new virtual environment. Note that the Python version needs to be at least 3.6 for everything to run. 
The default might be a lower version so if that is the case you must make sure that the correct version is used.
You can pass the correct python version to virtualenv via the `-p` parameter, for example
    ```bash
    $ virtualenv -p python3.6 <envname>
    ```
* Activate the environment via
    ```bash
    $ source <envname>/bin/activate
    ```
* Install the required packages (see section "Prerequisites"). Don't forget **cupy**, **tb_chainer** and **tensorflow**!
* Prepare the training data
or transfer existing data on the server
* To avoid stopping the training after disconnecting from the server, you might want to use a terminal-multiplexer 
such as [tmux](https://wiki.ubuntuusers.de/tmux/) or [screen](https://wiki.ubuntuusers.de/Screen/)
* Set the CUDA_PATH and LD_LIBRARY_PATH variables if they are not already set. The command should be something like
    ```bash
    $ export CUDA_PATH=/usr/local/cuda
    $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    ```
* To download the ResNet-50 caffemodel (it isn't downloaded automatically) see [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777) and save it where necessary (an error will tell you where if you try to create a TextLocEnv).
* Start training!

These instructions are for starting from scratch, for example if there is already a suitable virtual environment you 
obviously don't need to create a new one.
