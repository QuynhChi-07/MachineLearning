from retail_project.connectors.employee_connector import EmployeeConnector

ec=EmployeeConnector()
ec.connect()
em=ec.login("putin@hotmail.com", "123")
if em==None:
    print("Login Failed")
else:
    print("Login Succeeded")
    print (em)

#Test get all employee
print ("List of employee")
ds=ec.get_all_employee()
print(ds)
for emp in ds:
    print(emp)

id=3
emp=ec.get_detail_infor(id)
if emp==None:
    print("Không có nhân viên nào có mã =", id)
else:
    print ("Tìm thấy nhân vân có mã =", id)
    print(emp)