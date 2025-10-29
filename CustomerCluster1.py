import numpy as np
import pymysql
import pandas as pd
import traceback

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import plotly.express as px

def getConnect(server, port, database, username, password):
    try:
        conn = pymysql.connect(
            host=server,
            port=port,
            user=username,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor  # để trả về dạng dict, dễ đọc
        )
        print("Kết nối MySQL thành công!")
        return conn
    except Exception as e:
        print("Lỗi kết nối MySQL:")
        traceback.print_exc()
        print(e)
        return None

def closeConnection(conn):
    if conn:
        conn.close()
        print("Đã đóng kết nối MySQL.")

def queryDataset(conn, sql):
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            return df
    except Exception as e:
        print("Lỗi khi truy vấn dữ liệu:")
        traceback.print_exc()
        print(e)
        return pd.DataFrame()

conn = getConnect('localhost', 3306, 'salesdatabase', 'root', '@Obama123')
if conn:
        sql1 = "SELECT * FROM customer"
        df = queryDataset(conn, sql1)
        print(df)

sql2= """
        SELECT DISTINCT c.CustomerID, c.Age, cs.Annual_Income, cs.Spending_Score
        FROM customer c
        JOIN customer_spend_score cs ON c.CustomerID = cs.CustomerID
    """
df2 = queryDataset(conn, sql2)
df2.columns = ['CustomerID', 'Age', 'Annual_Income', 'Spending_Score']
print(df2)
print(df2.head())
print(df2.describe())

#histogram tính cho Age, Annual Income, Spending Score
def showHistogram(df,columns):
    plt.figure(1,figsize = (7,8))
    n = 0
    for column in columns:
        n+=1
        plt.subplot(3,1,n)
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        sns.distplot(df[column], bins = 32)
        plt.title(f"Histogram of {column}")
    plt.show()
showHistogram(df2,df2.columns[1:])

#Thiết kế hàm elbowMethod
def elbowMethod(df,columnsForElbow):
    X=df.loc[:,columnsForElbow].values
    inertia =[]
    for n in range (1,11):
        model = KMeans(n_clusters=n,
                       init="k-means++",
                       max_iter=500,random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)
    plt.figure(1,figsize = (15,6))
    plt.plot(range(1,11),inertia,'o')
    plt.plot(range(1,11),inertia,'-',alpha = 0.5)
    plt.xlabel("Number of clusters") ,plt.ylabel("Cluster sum of squared distance")
    plt.show()
columns =["Age","Spending_Score"]
elbowMethod(df2,columns)

#gom, cluster theo elbow => k=4
def runKMeans(X, cluster):
    model=KMeans(n_clusters=cluster,init='k-means++',max_iter=500,random_state=42)
    model.fit(X)
    labels=model.labels_
    centroids=model.cluster_centers_
    y_kmeans=model.fit_predict(X)
    return y_kmeans, centroids, labels

X=df2.loc[:, columns].values
cluster=4
colors=["red", "green", "blue", "purple", "black", "pink", "orange"]

y_kmeans, centroids, labels = runKMeans(X, cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2["cluster"]=labels

#trực quan hoá kết quả gom cụm
def visualizeKMeans(X, y_kmeans, cluster, title,xlabel,ylabel, colors):
    plt.figure(figsize = (10,10))
    for i in range (cluster):
        plt.scatter(X[y_kmeans==i, 0],X[y_kmeans==i, 1],s=100, c=colors[i], label='Cluster %d'%(i+1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
visualizeKMeans(X,y_kmeans,cluster,"Cluster of Customers - Age X Spending Score","Age","Spending Score",colors)

#Chạy k=5
columns = ['Annual_Income', 'Spending_Score']
elbowMethod(df2,columns)
X=df2.loc[:, columns].values
cluster=5

y_kmeans, centroids, labels = runKMeans(X, cluster)

print(y_kmeans)
print(centroids)
print(labels)
df2["cluster"]=labels

visualizeKMeans(X, y_kmeans,cluster,"Cluster of Customer - Annual Income X Spending Score", "Annual Income","Spending Score",colors)

#Chạy k=6
columns = ['Age','Annual_Income', 'Spending_Score']
elbowMethod(df2,columns)
X=df2.loc[:, columns].values
cluster=6

y_kmeans, centroids, labels = runKMeans(X, cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2["cluster"]=labels
print(df2)

def visualize3DKMeans(df, columns, hover_data, cluster):
    fig=px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2],color='cluster', hover_data=hover_data, category_orders={"cluster": range(0, cluster)})
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
hover_data=df2.columns
visualize3DKMeans(df2, columns, hover_data, cluster)

#Hàm in danh sách Customer theo cụm ra console
import webbrowser

# Hiện danh sách khách hàng theo cụm trên CONSOLE
def printCustomerByCluster(df):
    """
    In danh sách chi tiết của các Customer tương ứng với từng cụm ra console.
    """
    if 'cluster' not in df.columns:
        print("Chưa có cột 'cluster' trong DataFrame!")
        return

    clusters = sorted(df['cluster'].unique())
    print(f"\n==========================")
    print(f"TỔNG SỐ CỤM: {len(clusters)}")
    print("==========================")

    for c in clusters:
        print(f"CỤM {c}")
        subset = df[df['cluster'] == c]
        print(subset[['CustomerID', 'Age', 'Annual_Income', 'Spending_Score', 'cluster']])
        print(f"Số lượng khách hàng trong cụm {c}: {len(subset)}")

# Gọi thử
printCustomerByCluster(df2)

# Xuất danh sách khách hàng theo cụm ra WEB (HTML)
import webbrowser

def exportCustomerByClusterToWeb(df, filename="customer_clusters.html"):
    """
    Xuất danh sách chi tiết của các Customer theo cụm ra giao diện web (HTML file, hỗ trợ tiếng Việt chuẩn).
    """
    if 'cluster' not in df.columns:
        print("Chưa có cột 'cluster' trong DataFrame!")
        return

    # Tạo nội dung HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <title>Chi tiết cụm khách hàng</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 30px; background: #f7f7f7;}
            h1 {text-align: center; color: #333;}
            h2 {color: #2a7cbb; margin-top: 40px;}
            table {width: 100%; border-collapse: collapse; margin-bottom: 20px;}
            th, td {border: 1px solid #aaa; padding: 8px; text-align: center;}
            th {background-color: #2a7cbb; color: white;}
            tr:nth-child(even) {background-color: #f2f2f2;}
            hr {border: 1px solid #ccc; margin-top: 40px;}
        </style>
    </head>
    <body>
        <h1>Chi tiết cụm khách hàng</h1>
    """

    clusters = sorted(df['cluster'].unique())
    for c in clusters:
        html_content += f"<h2>Cụm {c}</h2>"
        subset = df[df['cluster'] == c]
        html_content += subset[['CustomerID', 'Age', 'Annual_Income', 'Spending_Score', 'cluster']].to_html(index=False, border=1)
        html_content += f"<p><b>Số lượng khách hàng trong cụm {c}:</b> {len(subset)}</p><hr>"

    html_content += """
    </body>
    </html>
    """

    # CHỈ KHÁC Ở ĐÂY: Dùng utf-8-sig để đảm bảo hiển thị tiếng Việt đúng
    with open(filename, "w", encoding="utf-8-sig") as f:
        f.write(html_content)

    print(f"Đã xuất danh sách chi tiết khách hàng theo cụm ra file: {filename}")
    webbrowser.open(filename)

# Gọi thử
exportCustomerByClusterToWeb(df2)


