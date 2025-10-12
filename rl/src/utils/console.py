from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def print_experiment_header(cfg, ctx):
    """Pretty console header like CleanRL / Lightning style with better spacing."""
    meta_table = Table.grid(padding=(0, 2))
    meta_table.add_row(f"[bold cyan]Experiment[/]: {ctx.exp_id}")
    meta_table.add_row(f"[bold cyan]Started[/]: {ctx.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    meta_table.add_row(f"[bold cyan]Log Dir[/]: {str(ctx.log_dir)}")

    console.print(
        Panel(
            meta_table,
            title="[bold green]Experiment Initialized[/]",
            expand=False,
            padding=(1, 2),
            border_style="green"
        )
    )

    cfg_table = Table(
        title="[bold yellow]Config Summary",
        show_lines=True,
        expand=True,
        pad_edge=True,
        padding=(0, 1),
    )
    cfg_table.add_column("Section", justify="right", style="bold cyan", no_wrap=True)
    cfg_table.add_column("Values", justify="left")

    dataset_summary = f"env={cfg.dataset.env}\nchallenge={cfg.dataset.challenge}\nrtg_scale={cfg.dataset.rtg_scale}"
    model_summary = f"name={cfg.model.name}\ncontext_len={cfg.model.context_len}\nembed_dim={cfg.model.embed_dim}"
    train_summary = (
        f"batch={cfg.training.batch_size}\nlr={cfg.optimizer.lr}\n"
        f"warmup={cfg.scheduler.warmup_steps}\nmax_iter={cfg.training.max_iters}"
    )

    cfg_table.add_row("Dataset", dataset_summary)
    cfg_table.add_row("Model", model_summary)
    cfg_table.add_row("Training", train_summary)

    console.print(cfg_table)
    console.rule("[bold green]Logging Started", style="green")
