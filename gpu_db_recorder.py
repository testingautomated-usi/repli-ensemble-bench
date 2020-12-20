import json
import os
import sqlite3
import threading
import time

import systemmonitor

DATABASE = 'database.db'
EVENTS_FOLDER = 'events'


def get_db():
    db = sqlite3.connect(DATABASE)
    db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                timestmp FLOAT NOT NULL,
                description TEXT NOT NULL,
                model INTEGER
            )
            """)
    db.execute("""
            CREATE TABLE IF NOT EXISTS gpu (
                mean_time FLOAT NOT NULL,
                id_on_system INTEGER NOT NULL,
                identifier VARCHAR NOT NULL,
                util FLOAT NOT NULL,
                memory_abs FLOAT NOT NULL,
                memory_util FLOAT NOT NULL,
                temp FLOAT NOT NULL
            )
            """)
    db.commit()
    return db


class Event:
    def __init__(self,
                 desc: str,
                 model: int):
        self.desc = desc
        self.model = model
        self.timestamp = time.time()


def dump(event):
    path = os.path.join(EVENTS_FOLDER, f"{event.timestamp}.json")
    with open(path, 'w') as outfile:
        json.dump(event.__dict__, outfile)


def add_events_to_db():
    database = get_db()
    for filename in os.listdir(EVENTS_FOLDER):
        if filename.endswith(".json"):
            full_path = os.path.join(EVENTS_FOLDER, filename)
            with open(full_path) as json_file:
                event = json.load(json_file)
                query = (f"INSERT INTO events VALUES (" \
                         f"{event['timestamp']}, " \
                         f"'{event['desc']}', " \
                         f"{event['model']})")
                database.execute(query)
                database.commit()
        elif not filename.endswith('.gitkeep'):
            raise RuntimeWarning(f"Unknown file ending {filename}")


def get_gpu_observer(database):
    def add_gpu_state_to_db(state: systemmonitor.DeviceState) -> None:
        database.execute((f"INSERT INTO gpu VALUES ("
                          f"{state.mean_time}, "
                          f"{state.index}, "
                          f"'{state.uuid}', "
                          f"{state.gpu_util}, "
                          f"{state.mem_used}, "
                          f"{state.mem_util}, "
                          f"{state.temp})"))
        database.commit()

    return add_gpu_state_to_db


def record():
    db = get_db()
    systemmonitor.run([get_gpu_observer(db)], average_window_sec=60)


class BackgroundMonitoring(object):
    def __init__(self):
        thread = threading.Thread(target=record, args=())
        thread.daemon = True
        thread.start()
        self.thread = thread

    def stop(self):
        systemmonitor.exit_event.set()
        print("Gave monitoring exit signal. Waiting for thread to join.")
        self.thread.join()
        print("Background Monitoring Thread joined")
