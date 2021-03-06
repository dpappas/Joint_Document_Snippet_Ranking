
# A Neural Model for Joint Document and Snippet Ranking in Question Answering for Large Document Collections

This is the implementation of ["A Neural Model for Joint Document and Snippet Ranking in Question Answering for Large Document Collections"](http://google.com) in Pytorch presented in ACL IJCNLP 2021.

### File Overview

### Data Processing

For Pubmed indexing we used ElasticSearch v5.0.1 with Lucene v6.2.1 

For Modified Natural Questions indexing we used ElasticSearch v7.10.0 with Lucene v8.7.0
First you need to download the [official dataset](https://ai.google.com/research/NaturalQuestions/download).
 
##### PubMED 

All PubMed data were collected from the official repository:
https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ 

Updates performed weekly using data from the official update repository:
https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/

##### Natural Questions

The Natural Questions dataset is too big to share on GitHub. 
Instead we share the code to preprocess the Natural Questions data.
You can find the code in the direcory: `NaturalQuestions/data_processing/` 

### Example commands 

In order to run properly the extraction files you need to change the following paths:
```
Path to the w2v embeddings file: Variable w2v_bin_path
Path to the precomputed idf file: Variable idf_pickle_path
Path to the output directory: Variable odir
Path to the 1st evaluation script: Variable eval_path
Path to the 2nd evaluation script: Variable retrieval_jar_path
Path to the pretrained model: Variable resume_from
```

For The Natural Questions dataset you have to change the paremeters above and then execute
```
python3.6 extract_w2v_jpdrmm.py
```

For the BioASQ dataset you have to change the paremeters above and then execute
```
python3.6 extract_from_trained_ablation.py 1111111
```
The `resume_from` parameter has to be one of the directories in the `./trained_models/ablation_models/` directory 

### Author

* [Dimitris Pappas](dpappas@aueb.gr)

### Cite

If you find our work useful please cite our paper

```
@inproceedings{pappas-androutsopoulos-2021-neural,
    title = "A Neural Model for Joint Document and Snippet Ranking in Question Answering for Large Document Collections",
    author = "Pappas, Dimitris  and Androutsopoulos, Ion",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.301",
    doi = "10.18653/v1/2021.acl-long.301",
    pages = "3896--3907"
}
```


