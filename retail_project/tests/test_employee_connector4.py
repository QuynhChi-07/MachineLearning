from retail_project.connectors.employee_connector import EmployeeConnector
from retail_project.models.employee import Employee

ec=EmployeeConnector()
ec.connect()
emp=Employee()
emp.ID=7
# emp.EmployeeCode="EMP111"
# emp.Name="FKT"
# emp.Phone="022334455"
# emp.Email="fkt@gmail.com"
# emp.Password="123"
# emp.IsDeleted=0

result=ec.delete_one_employee(emp)
if result >0:
    print("delete success")
else:
    print("delete fail")
