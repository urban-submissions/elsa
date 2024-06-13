from __future__ import annotations

import sqlite3

import pandas as pd

import magicpandas as magic
from elsa.resource import Resource
# import elsa.sql as sql
import elsa.sql as sql

if False:
    from .annotation import Annotation


# todo: where to keep sql table file?
class SQL(Resource):
    __outer__: Annotation
    database: str = "annotation.sqlite"
    name: str = "annotation"


    def push(self):
        connect = sqlite3.connect(self.database)
        c = connect.cursor()
        c.execute(sql.annotation.create)
        c.execute(sql.annotation.clear)
        ann = self.__outer__
        ann.to_sql(
            self.name,
            connect,
            if_exists="append",
            dtype={
                'iann': 'int',
                'file': 'text',
                'data_source': 'text',
                'label': 'text',
                'w': 'float',
                's': 'float',
                'e': 'float',
                'n': 'float',
            }
        )

    def pull(self) -> pd.DataFrame:
        connect = sqlite3.connect(self.database)
        result = pd.read_sql_table(self.name, connect)
        return result
