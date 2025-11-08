import pymysql
import traceback
import pandas as pd

class Connector:
    def __init__(self, server=None, port=None, database=None, username=None, password=None):
        self.server = server
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.conn = None

    def connect(self):
        try:
            self.conn = pymysql.connect(
                host=self.server,
                port=int(self.port),
                user=self.username,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor
            )
            return self.conn
        except Exception as e:
            self.conn = None
            print("Database connection failed:", e)
            traceback.print_exc()
            return None

    def disConnect(self):
        try:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
        except Exception as e:
            print("Error disconnecting:", e)
            traceback.print_exc()

    def queryDataset(self, sql):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                df = pd.DataFrame(cursor.fetchall())
                if not df.empty:
                    df.columns = df.columns  # tên cột giữ nguyên do DictCursor
                return df
        except Exception as e:
            print("Query failed:", e)
            traceback.print_exc()
            return None

    def getTablesName(self):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SHOW TABLES;")
                results = cursor.fetchall()
                tablesName = [list(item.values())[0] for item in results]
                return tablesName
        except Exception as e:
            print("Could not get tables:", e)
            traceback.print_exc()
            return []
