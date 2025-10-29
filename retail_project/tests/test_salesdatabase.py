from retail_project.connectors.connector import Connector

conn=Connector(database='salesdatabase')
conn.connect()
sql="select * from customer"
df=conn.queryDataset(sql)
print(df)

