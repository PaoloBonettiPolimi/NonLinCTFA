import subprocess

# change n_feats

subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "50", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "75", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "150", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "200", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "300", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "400", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment2.py", "--noise", "7.5", "--n_feats", "500", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
