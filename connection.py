
import pyodbc
 
def create_connection(server, database, username, password):

    """ Create a database connection to the SQL Server database """

    conn = None

    try:

        conn = pyodbc.connect(

            'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password

        )

        print("Connection established.")

    except pyodbc.Error as e:

        print(e)

    return conn
 
def insert_user(conn, name, age, email):

    """ Insert a new user into the users table """

    try:

        cursor = conn.cursor()

        cursor.execute("INSERT INTO users (name, age, email) VALUES (?, ?, ?)", (name, age, email))

        conn.commit()

        print("User added successfully.")

    except pyodbc.Error as e:

        print(e)
 
def main():

    server = "192.168.137.7"

    database = "MBSPL_CrackDetection"

    username = "amits"

    password = "amit12345"

    conn = create_connection(server, database, username, password)

    if conn is not None:

        while True:

            name = input("Enter user's name (or 'quit' to exit): ")

            if name.lower() == 'quit':

                break

            age = int(input("Enter user's age: "))

            email = input("Enter user's email: ")

            insert_user(conn, name, age, email)

        conn.close()

    else:

        print("Error! Cannot create the database connection.")
 
if __name__ == '__main__':

    main()
