import subprocess

# change n_tasks

subprocess.call(["python", "DdimensionalExperiment4.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "2", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment4.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "5", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment4.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment4.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "20", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment4.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "50", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment4.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "100", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
