import sqlite3
from math import floor

ROUND_DIGITS = 2

EXPERIMENTS = {
    'cifar_ubu_2xgpu': ['mainprocess', 'multiprocess', 'multigpu'],
    'cifar_10_on_win_2080_ti': ['mainprocess', 'multiprocess'],
}


def _fetch_one_row_one_value_save(db: sqlite3.Connection, query: str):
    results = db.execute(query).fetchmany(2)
    assert len(results) == 1
    return results[0][0]


if __name__ == '__main__':
    for run, contexts in EXPERIMENTS.items():
        db = sqlite3.connect(f"results/{run}/database.db")

        print("\n\n")
        print(f"### PC: {run}")

        for context in contexts:
            start_end_query = "select timestmp from events where description like '{} % {}'"
            ensemble_start = _fetch_one_row_one_value_save(db, start_end_query.format("start", context))
            ensemble_end = _fetch_one_row_one_value_save(db, start_end_query.format("end", context))

            print('')
            print(f'Context: {context}')
            hours = floor((ensemble_end - ensemble_start) / 3600)
            minutes = floor(((ensemble_end - ensemble_start) - (hours * 3600)) / 60)
            print(f">\tTotal training time: {hours}h {minutes}min<br>")

            device_loads = db.execute(
                """ 
                SELECT id_on_system, avg(util)
                FROM gpu
                where mean_time > {}
                and mean_time < {}
                GROUP BY id_on_system
                ORDER BY id_on_system asc 
                """.format(ensemble_start, ensemble_end)
            ).fetchall()

            print(f">\tAvg load CPU: {round(device_loads[0][1], ROUND_DIGITS)}%<br>")
            for i in range(1, len(device_loads)):
                print(f">\tAvg load GPU {i - 1}: {round(device_loads[i][1], ROUND_DIGITS)}%<br>")
