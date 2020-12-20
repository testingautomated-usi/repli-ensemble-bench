import asyncio
import os
import pickle
import shutil
import time

import case_study_cifar10
import gpu_db_recorder
import uncertainty_wizard as uwiz

temp_dir = "F:\\temp\\ensemble"


def ensembles():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Train using multiprocessing
    gpu_db_recorder.dump(gpu_db_recorder.Event("start experiment multiprocess", model=-1))
    ensemble = uwiz.models.LazyEnsemble(num_models=20, model_save_path=temp_dir, delete_existing=True,
                                        default_num_processes=5)
    history = ensemble.create(case_study_cifar10.train_model)
    gpu_db_recorder.dump(gpu_db_recorder.Event("end experiment multiprocess", model=-1))
    with open('history_multiprocess.pickle', 'wb+') as f:
        pickle.dump(history, f)
    shutil.rmtree(temp_dir)

    # Train on main process
    gpu_db_recorder.dump(gpu_db_recorder.Event("start experiment mainprocess", model=-1))
    ensemble = uwiz.models.LazyEnsemble(num_models=20, model_save_path=temp_dir, delete_existing=True,
                                        default_num_processes=0)
    history = ensemble.create(case_study_cifar10.train_model)
    gpu_db_recorder.dump(gpu_db_recorder.Event("end experiment mainprocess", model=-1))
    with open('history_mainprocess.pickle', 'wb+') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    print("Start Monitoring")
    monitoring = gpu_db_recorder.BackgroundMonitoring()

    print("Init Training")
    ensembles()

    print("Done Training. Stop Monitoring")
    monitoring.stop()

    gpu_db_recorder.add_events_to_db()


