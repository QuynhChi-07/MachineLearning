#python -m pip install mysql-connector-python
import pymysql
import traceback
import pandas as pd
class Connector:
    def __init__(self,server="localhost", port=3306, database="k23416_retaiil", username="root", password="@Obama123"):
        self.server=server
        self.port=port
        self.database=database
        self.username=username
        self.password=password
    def connect(self):
        try:
            self.conn = pymysql.connect(
                host=self.server,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password)
            return self.conn
        except:
            self.conn=None
            traceback.print_exc()
        return None

    def disConnect(self):
        if self.conn != None:
            self.conn.close()

    def queryDataset(self, sql):
        """
        Chạy câu SQL và trả về kết quả dưới dạng pandas DataFrame
        """
        try:
            if not self.conn:
                raise ValueError("Chưa có kết nối database. Hãy gọi connect() trước.")

            df = pd.read_sql(sql, self.conn)
            return df

        except Exception as e:
            print("Lỗi khi chạy queryDataset:", e)
            traceback.print_exc()
            return pd.DataFrame()
    def getTablesName(self):
        cursor = self.conn.cursor()
        cursor.execute("Show tables;")
        results=cursor.fetchall()
        tablesName=[]
        for item in results:
            tablesName.append([tableName for tableName in item][0])
        return tablesName
    def fetchone(self,sql,val):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,val)
            dataset=cursor.fetchone()
            cursor.close()
            return dataset
        except:
            traceback.print_exc()
        return None
    def fetchall(self,sql,val):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,val)
            dataset=cursor.fetchall()
            cursor.close()
            return dataset
        except:
            traceback.print_exc()
        return None
    def insert_one(self, sql, val):
        cursor=self.conn.cursor()
        cursor.execute(sql,val)
        self.conn.commit()
        result=cursor.rowcount
        cursor.close()
        return result