## IMPORTANT - If you are a collaborator:

There are steps to follow before you run any code so that the data lives in an accessible place for all of us:

- Go to Google Drive and navigate to **Shared with me**
- Right-click on the folder that has been shared with you.
- Select **Organize** -> **Add shortcut**
- Choose **All locations** -> **My Drive** and click **Add**

## How to use this repository:

The setup code in this repository is designed to be run on Google Colab. IF YOU DO NOT ALREADY HAVE THE CSVs 
AND EMBEDDINGS, please follow the steps below to create a csv of the MPD data and create your embeddings with
your encoding model of choice:

- Create a Google Colab notebook **with a GPU Runtime** if possible, and run the following code in the first and second cells:

```python
import sys
import importlib
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/siddmohanty111/PlaylistClustering.git

%cd /content/PlaylistClustering
!pip install -r requirements.txt
!pip install sentence-transformers
```

```shell
!PYTHONPATH=/content/PlaylistClustering:$PYTHONPATH
!python firsttimesetup.py
```

- When you are finished with your work on the notebook, navigate to File -> Save a copy on GitHub

## Acknowledgements

Portions of the code and concepts in this repository were adapted from the [LLM-Playlist-Recommender](https://github.com/elea-vellard/LLM-Playlist-Recommender) project by Elea Vellard. We thank them for their open-source contributions.

