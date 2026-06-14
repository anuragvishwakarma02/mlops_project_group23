"""Shared Weights & Biases initialization helpers."""

import os

import wandb


def init_wandb_run(project, name, config=None, resume=None):
    """Initialize a W&B run using WANDB_API_KEY from environment."""
    api_key = os.getenv('WANDB_API_KEY')

    init_kwargs = {
        'project': project,
        'name': name,
    }
    if config is not None:
        init_kwargs['config'] = config
    if resume is not None:
        init_kwargs['resume'] = resume

    if api_key:
        wandb.login(key=api_key)
        enabled = True
    else:
        init_kwargs['mode'] = 'disabled'
        enabled = False

    wandb.init(**init_kwargs)
    return enabled
