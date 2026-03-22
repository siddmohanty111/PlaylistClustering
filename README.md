## If you are a collaborator:

There are steps to follow before you run any code so that the data lives in an accessible place for all of us:

- Go to Google Drive and navigate to **Shared with me**
- Right-click on the folder that has been shared with you.
- Select **Organize** -> **Add shortcut**
- Choose **All locations** -> **My Drive** and click **Add**

## How to use this repository:

The setup code in this repository is designed to be run on Google Colab. IF YOU DO NOT ALREADY HAVE THE CSVS 
AND EMBEDDINGS, please follow the steps below to create a csv of the MPD data and create your embeddings with
your encoding model of choice:

- Create a Google Colab notebook, and run the following code in the first cell:

```python
import sys
import importlib
from google.colab import drive

!git clone https://github.com/siddmohanty111/PlaylistClustering.git

%cd /content/PlaylistClustering
!pip install -r requirements.txt
!pip install sentence-transformers
```

```shell
!python parsedata.py
```

- Run json_to_csv.ipynb (GPU Runtime not needed). This will turn the raw MPD json data into csv form
- Run create_embeddings.ipynb (GPU Runtime necessary for quick embedding generation). This uses Sentence BERT (all-MiniLM-L6-v2 by default) to create embeddings. 
- When you are finished with your work on the notebook, navigate to File -> Save a copy on GitHub