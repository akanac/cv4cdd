We use the save event log collection as in the paper by Adams, Jan Niklas, et al. "An Experimental Evaluation of Process Concept Drift Detection." Proceedings of the VLDB Endowment 16.8 (2023): 1856-1869.
The used event log collection is available in the folder "EvaluationLogs" in the paper's git repository: https://github.com/cpitsch/cdrift-evaluation/tree/main/EvaluationLogs

Following the same procedure as used in the evaluation scritp of the Adams' repository, we excluded:
* event logs from Ostovar dataset that are not atomic,
* event logs from Ceravolo dataset that contain less than 1000 traces.
