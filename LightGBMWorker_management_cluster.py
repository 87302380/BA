import os
import pickle
import logging
import argparse
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
import LightGBMWorker as worker

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)

logger.addHandler(fh)


def get_parameters(selected_x, selected_y, kFold, num_threads):
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=1)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=100)
    # parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--shared_directory', type=str,help='A directory that is accessible for all processes, e.g. a NFS share.', default='./result')
    parser.add_argument('--nic_name', type=str, default='lo')

    args = parser.parse_args()
    # if args.worker:
    #     import time
    #     time.sleep(5)   # short artificial delay to make sure the nameserver is already running
    #     w = worker(data)
    #
    #    # w.load_nameserver_credentials(working_directory=args.shared_directory)
    #     w.run(background=False)
    #     exit(0)

    host = hpns.nic_name_to_host(args.nic_name)

    result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)

    # Step 1: Start a nameserver
    # Every run needs a nameserver. It could be a 'static' server with a
    # permanent address, but here it will be started for the local machine with the default port.
    # The nameserver manages the concurrent running workers across all possible threads or clusternodes.
    # Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
    NS = hpns.NameServer(run_id='example1', host=host, port=0)
    ns_host, ns_port = NS.start()

    # Step 2: Start a worker
    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.
    w = worker(selected_x, selected_y, kFold, num_threads, host=host, run_id='example1', nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)
    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # Here, we run BOHB, but that is not essential.
    # The run method will return the `Result` that contains all runs performed.
    bohb = BOHB(configspace=w.get_configspace(),
                run_id='example1',
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                result_logger=result_logger,
                min_budget=args.min_budget, max_budget=args.max_budget
                )
    res = bohb.run(n_iterations=args.n_iterations)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    info = res.get_runs_by_id(incumbent)

    parameter = id2config[incumbent]['config']
    min_error = info[0]['loss']
    #booster = info[0]['info']

    return parameter, min_error#, booster
