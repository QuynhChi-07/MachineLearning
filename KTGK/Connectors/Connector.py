import pymysql
import traceback
import pandas as pd

class Connector:
    def __init__(self,
                 server="localhost",
                 port=3306,
                 database="retails",
                 username="root",
                 password="@Obama123"):
        self.server = server
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.conn = None

    def connect(self):
        """Tạo kết nối tới MySQL"""
        try:
            self.conn = pymysql.connect(
                host=self.server,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                # THAY ĐỔI QUAN TRỌNG: Xóa dòng DictCursor để dùng Cursor mặc định (trả về tuple)
                # cursorclass=pymysql.cursors.DictCursor
            )
            print("Kết nối database thành công!")
            return self.conn
        except Exception:
            self.conn = None
            print("Lỗi khi kết nối database!")
            traceback.print_exc()
            return None

    def disConnect(self):
        """Đóng kết nối"""
        if self.conn:
            self.conn.close()
            print("Đã đóng kết nối database.")

    # SỬA ĐỔI: Thêm tham số 'params=None' và truyền nó vào pd.read_sql
    def queryDataset(self, sql, params=None):
        """Chạy câu SQL và trả về DataFrame"""
        try:
            if not self.conn:
                # Thay đổi ValueError thành Exception nếu bạn muốn traceback chi tiết hơn
                raise Exception("Chưa có kết nối database. Hãy gọi connect() trước.") 

            # Truyền tham số 'params' vào pd.read_sql để hỗ trợ truy vấn tham số hóa
            df = pd.read_sql(sql, self.conn, params=params) 
            return df

        except Exception as e:
            print("Lỗi khi chạy queryDataset:", e)
            traceback.print_exc()
            return pd.DataFrame()

    def getTablesName(self):
        """Lấy danh sách bảng trong DB"""
        try:
            # Nếu dùng DictCursor, cần lấy tên khóa đầu tiên để có tên bảng
            with self.conn.cursor() as cursor:
                cursor.execute("SHOW TABLES;")
                tables = [row[next(iter(row))] for row in cursor.fetchall()]
                return tables
        except Exception:
            traceback.print_exc()
            return []

    def fetchone(self, sql, val=None):
        """Lấy 1 bản ghi"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, val)
                return cursor.fetchone()
        except Exception:
            traceback.print_exc()
            return None

    def fetchall(self, sql, val=None):
        """Lấy nhiều bản ghi"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, val)
                return cursor.fetchall()
        except Exception:
            traceback.print_exc()
            return []

    def insert_one(self, sql, val):
        """Thực thi INSERT/UPDATE/DELETE"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, val)
                self.conn.commit()
                return cursor.rowcount
        except Exception:
            traceback.print_exc()
            self.conn.rollback()
            return 0