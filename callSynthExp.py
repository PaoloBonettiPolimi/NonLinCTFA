import subprocess

# basic configuration
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])

### change train dim

subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "100", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "150", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "200", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "300", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "400", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "500", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])

# change n_feats

subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "50", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "75", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "150", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "200", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "300", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "400", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "500", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])

# change noise

subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "2.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "10", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "15", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])

# change n_tasks

subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "2", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "5", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "10", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "20", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "50", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
subprocess.call(["python", "DdimensionalExperiment.py", "--noise", "7.5", "--n_feats", "100", "--n_tasks", "100", "--train_dim", "250", "--eps1", "0", "--eps2", "0.0001", "--n_reps", "10"])
