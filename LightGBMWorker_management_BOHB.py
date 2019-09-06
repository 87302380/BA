import logging
import argparse
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from LightGBMWorker import LightGBMWorker as worker

logging.basicConfig(level=logging.WARNING)

from ConfigSpace.read_and_write import json


def get_parameters(train_data, kFold, iterations):
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=1)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=iterations) # max value = 4
    # parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--shared_directory', type=str,help='A directory that is accessible for all processes, e.g. a NFS share.', default='./result')
    # parser.add_argument('--nic_name', type=str, default='lo')
    args = parser.parse_args()



    result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)

    NS = hpns.NameServer(run_id='BOHB', host='127.0.0.1', port=None)
    NS.start()

    w = worker(train_data, kFold, nameserver='127.0.0.1', run_id='BOHB')
    w.run(background=True)

    with open(args.shared_directory + "/configspace.json", 'w') as fh:
        fh.write(json.write(w.get_configspace()))

    bohb = BOHB(configspace=w.get_configspace(),
                run_id='BOHB', nameserver='127.0.0.1',
                min_budget=args.min_budget, max_budget=args.max_budget,
                result_logger=result_logger
                )
    res = bohb.run(n_iterations=args.n_iterations)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    info = res.get_runs_by_id(incumbent)


    parameter = id2config[incumbent]['config']
    min_error = info[0]['loss']

    return parameter, min_error
