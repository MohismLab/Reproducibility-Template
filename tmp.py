from your_project.utils import parse_args_and_update_config, create_experiment_snapshot


config = parse_args_and_update_config()

commit_hash = create_experiment_snapshot()
config.git_commit_hash = commit_hash

print(config)
