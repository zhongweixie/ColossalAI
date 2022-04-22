import click
from .run import launch_multi_processes
from colossalai.context import Config


@click.command(help="Launch distributed training on a single node or multiple nodes",
               context_settings=dict(ignore_unknown_options=True))
@click.option("-H", "-host", "--host", type=str, default=None, help="the list of machines to launch")
@click.option("--hostfile",
              type=str,
              default=None,
              help="Hostfile path that defines the device pool available to the job (e.g. worker-name:number of slots)")
@click.option(
    "--include",
    type=str,
    default=None,
    help=
    "Specify computing devices to use during execution. String format is NODE_SPEC@NODE_SPEC where NODE_SPEC=<worker-name>:<list-of-slots>"
)
@click.option(
    "--exclude",
    type=str,
    default=None,
    help=
    "Specify computing devices to NOT use during execution. Mutually exclusive with --include. Formatting is the same as --include."
)
@click.option("--num_nodes", type=int, default=-1, help="Total number of worker nodes to use.")
@click.option("--nproc_per_node", type=int, default=-1, help="Number of GPUs to use on each node.")
@click.option("--master_port",
              type=int,
              default=29500,
              help="(optional) Port used by PyTorch distributed for communication during distributed training.")
@click.option("--master_addr",
              type=str,
              default="127.0.0.1",
              help="(optional) IP address of node 0, will be inferred via 'hostname -I' if not specified.")
@click.option("--ssh-port", type=int, default=None, help="(optional) the port used for ssh connection")
@click.argument("user_script", type=str)
@click.argument('user_args', nargs=-1)
def run(host: str, hostfile: str, num_nodes: int, nproc_per_node: int, include: str, exclude: str, master_addr: str,
        master_port: int, ssh_port: int, user_script: str, user_args: str) -> None:
    """
    To launch multiple processes on a single node or multiple nodes via command line.

    Usage::
        # run on the current node with all available GPUs
        colossalai run train.py

        # run with only 2 GPUs on the current node
        colossalai run --nprocs_per_node 2 train.py

        # run on two nodes
        colossalai run --host <host1>,<host2> train.py

        # run with hostfile
        colossalai run --hostfile <file_path> train.py
    """
    if not user_script.endswith('.py'):
        click.echo(f'Error: invalid Python file {user_script}. Did you use a wrong option? Try colossalai run --help')
        exit()

    args_dict = locals()
    args = Config(args_dict)
    args.user_args = list(args.user_args)
    launch_multi_processes(args)
