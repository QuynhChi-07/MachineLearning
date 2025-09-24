import sqlite3
import pandas as pd
try:
    #Connect to DB and create a cursor
    sqliteConnection = sqlite3.connect('../databases/databases/Chinook_Sqlite.sqlite')
    cursor = sqliteConnection.cursor()
    print('DB Init')
    #Write a quuery and excute it with cursor
    query='SELECT * FROM InvoiceLine LIMIT 5;'
    cursor.execute(query)
    # Fetch data
    rows = cursor.fetchall()
    # Lấy tên cột từ cursor.description
    col_names = [description[0] for description in cursor.description]
    #Fetch and output result
    df = pd.DataFrame(rows, columns=col_names)
    print(df)
    #Close the cursor
    cursor.close()
#Handle errors
except sqlite3.Error as error:
    print('Error occured', error)
#Close DB Connection irrespective of success or failure
finally:
    if sqliteConnection:
        sqliteConnection.close()
        print('sqlite connection closed')

#output
"""
Cột 0 --> InvoiceLineID
...
"""

