# Machine-comprehension-Keras
Context based Question-Answering model based on NeuralQA paper for Stanford SQuAd Dataset.

# Dependencies
- keras 2.0 [tensorflow backend]
- tensorflow 1.0

# Running the code
To start training the model you must download the Stanford SQuAD datset from their website. Its a small json file (less than 50mb). Afterwards, executing the Save_Data.py will process data ready to be trained upon and save the data in .h5 files(won't take much time). Also the dictionary to convert words to vectors will be saved in .npy format. Once the data is ready Machine_Comprehens_Glove_WordEmbeddings.py can be executed. It will start 10 epochs on complete dataset having more than 119000 unique words. To get better results epochs must be increased but this model is computationaly very expensive even for 10 epochs. You need more than 8gb RAM for storing pre-trained glove embeddings of just 50 dimension. Increasing the dimension of Embeddings will further increase the memory required to train the model. I was unable to run this model for more than 1 epoch due to restricted memory and now I need to increase the memory to train it effecctively. Once I train the model I will upload the training file and the results.

Other notebooks are given for reference purpose like:
- How data is preprocessed
- How word and char embeddings are computed using CNN and LSTM (both of them had their pros and cons)
- How Glove embeddings are used rather than training the model to learn them from scratch.
- What was the loss after 1 epoch and how much time it took to complete just single epoch.

# Files
- CNN-char embeddings.ipynb: Contains code to compute CNN embeddings upon chars. This code can be employed with glove embeddings to increase accuracy, but will make the model even more computationaly expensive.
- Machine Comprehension (Glove word embeddings).ipynb: Code for the best model I was able to obtain till now after failing at various models.
- Machine Comprehension(LSTM-char embeddings).ipynb: An alternative model using char-embeddings rather than word embeddings, but not as effective as the final model.
- Save_data.py: Preprocessing and saving data to be used  afterwards for training.
- Screeenshot: Plot showing GPU's memory and graphics usage at training time. CPU usage is also provided.
- Others: Other files are rather .py files corresponding to their .ipynb files or the data files storing training data and dictionaries.

# GPU used for training:
Single Nvidia GTX 970

# Research paper referred
Making Neural QA as Simple as Possible but not Simpler
[Dirk Weissenborn, Georg Wiese, Laura Seiffe]
https://arxiv.org/pdf/1703.04816v2.pdf
