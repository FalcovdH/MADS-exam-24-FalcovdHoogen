Naam: Falco van den Hoogen
Studentnummer: 1867022


# MADS-exam-24
The junior datascientist at your work is pretty confident about his knowledge of all the models; He has helped you out by doing some data exploration for you, and he even created two models!

However, he didnt learn to hypertune things, and you are hired as a junior+ datascientist and he has heard you had pretty high grades in the machine learning course.
In addition to that, you are always telling weird stories about how vision models are the same as searching for undigested and vomited owl pellets in the forest, and he thinks you might be able to help him out.

His hopes are you are able to hypertune the models, but that you might also come up with some crazy creative ideas about how to improve the models.

## The data
We have two datasets:
### The PTB Diagnostic ECG Database

- Number of Samples: 14552
- Number of Categories: 2
- Sampling Frequency: 125Hz
- Data Source: Physionet's PTB Diagnostic Database

All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_train.parq` and `data/heart_test.parq`.

### Arrhythmia Dataset

- Number of Samples: 109446
- Number of Categories: 5
- Sampling Frequency: 125Hz
- Data Source: Physionet's MIT-BIH Arrhythmia Dataset
- Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_big_train.parq` and `data/heart_big_test.parq`.

## Exploration
In `notebooks/01_explore-heart.ipynb` you can find some exploration. It's not pretty, and
he hasnt had time to add a lot of comments, but he thinks it's pretty self explanatory for
someone with your skill level.

## Models
There are two notebooks with models. He has tried two approaches: a 2D approach, and a 1D approach. Yes, he is aware that this is a time series, but he has read that 2D approaches can work well for timeseries, and he has tried it out.

## Your task
You can find the models in `src/models.py`, but they are also in the notebooks themselves.

## 1. explore the models
Explore the existing models manually, for both datasets, to get some intuition about their performance
## 2. modify the models for hypertuning
Based on your knowlodge about hypertuning, you might want to change and expand the models: add more parameters you might want to tune, add additional layers, and if you feel like it you could also create additional models that you think might be promising. Try to balance exploit (improve what is already there) and explore (creativity/curiosity of new things).
Do some manual hypertuing to get a feel for the models and what might work, or what does not.
## 3. hypertune the models
When you are happy with your manual research, set up a hypertune.py file and hypertune the models with Ray. Make sure you log everything thats relevant for your model (eg with ray, mlflow of gin-config) including which dataset you use.
## 4. Analyse the hypertuning results
Create a notebook where you explore your hypertuning results, and save the plots that are most relevant for your research.
You can do steps 2-4 iteratively, so you might have some idea, do a rough gridsearch, tweak your model again, now search with some algorithm, etc.

## 5. Write a report
Create a short summary (2 pages) of your results.
Start with an overview of your top architectures (yes I know it hurts to leave out the models that failed).
Describe your searchspace, and show the relevant hyperparameters you have tuned.
You can take inspiration from table 3 in the (attention is all you need)[https://arxiv.org/pdf/1706.03762.pdf] paper for an excelent summary of hypertuning. Export your summary as pdf in your repo.

Because this is a medical dataset, an we are trying to spot disease, the client thinks it is more important that you find as much sick people as possible, even if that means you will have more false positives. So, they want you to optimize for recall, more than for precision.

When you are done, do two things:
0. Make your own private repo
1. invite https://github.com/raoulg to your repo as a collaborator
2. Upload the summary pdf to canvas

You will be graded for:
- Overall presentation and clarity of your work (organisation, comments, typehinting, etc). (10%)
- (step 2) The level of change and expansion of the models: balance performance (exploit) and creativity/curiosity (explore). (30%)
- (step 2 and 3) The relevance of your starting hyperparameters, based on your manual hypertuning. (10%)
- (step 3) The quality of your implementation of the hypertuning in your codebase (10%)
- (step 4 and 5) The clarity and relevance of your summary (40%)